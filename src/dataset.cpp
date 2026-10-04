#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Nvtx.hpp"
#include "DeepLearnLib/ParallelFor.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <pugixml.hpp>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

const std::vector<std::string> VOC_CLASSES_DEFAULT = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

namespace
{
constexpr int GRID_SIZE = 7;
constexpr int BOXES_PER_CELL = 2;
constexpr int BOX_PARAMS = 5;
// Two slots of five values (x, y, w, h, objectness); the class one-hot starts at index 10.
constexpr int CLASS_OFFSET = 10;
constexpr float NORMALIZATION_FACTOR = 255.0F;
constexpr float CENTER_DIVISOR = 2.0F;
constexpr int IMAGE_CHANNELS = 3;
constexpr int IMAGE_SIZE = 448;

// Converts VOC XML boxes into normalized YOLO labels (class, center, width, height).
// The loader then reads one .txt file per image, without parsing XML again.
auto convert_voc_to_yolo(const std::string& annot_dir, const std::string& label_dir, const std::string& jpeg_dir,
    const std::vector<std::string>& class_names) -> void
{
    std::unordered_map<std::string, int> class_map;
    for (size_t i = 0; i < class_names.size(); ++i)
    {
        class_map[class_names[i]] = static_cast<int>(i);
    }

    int converted_count = 0;
    for (const auto& directory_entry : std::filesystem::directory_iterator(annot_dir))
    {
        if (directory_entry.path().extension() != ".xml")
        {
            continue;
        }

        std::string base_name = directory_entry.path().stem().string();
        std::string xml_path = directory_entry.path().string();
        std::string text_path = label_dir + "/" + base_name + ".txt";
        std::string image_path = jpeg_dir + "/" + base_name + ".jpg";

        if (!std::filesystem::exists(image_path))
        {
            continue;
        }

        pugi::xml_document xml_doc;
        if (!xml_doc.load_file(xml_path.c_str()))
        {
            continue;
        }

        auto root_node = xml_doc.child("annotation");
        auto size_node = root_node.child("size");
        int image_width = size_node.child("width").text().as_int();
        int image_height = size_node.child("height").text().as_int();
        if (image_width == 0 || image_height == 0)
        {
            continue;
        }

        std::ofstream text_file(text_path);
        for (auto object_node = root_node.child("object"); object_node != nullptr;
             object_node = object_node.next_sibling("object"))
        {
            std::string class_name = object_node.child("name").text().as_string();
            auto bound_box = object_node.child("bndbox");
            if (bound_box.empty())
            {
                continue;
            }

            int class_id = -1;
            auto class_entry = class_map.find(class_name);
            if (class_entry != class_map.end())
            {
                class_id = class_entry->second;
            }

            if (class_id >= 0)
            {
                float x_min = bound_box.child("xmin").text().as_float();
                float y_min = bound_box.child("ymin").text().as_float();
                float x_max = bound_box.child("xmax").text().as_float();
                float y_max = bound_box.child("ymax").text().as_float();

                float x_center = ((x_min + x_max) / CENTER_DIVISOR) / static_cast<float>(image_width);
                float y_center = ((y_min + y_max) / CENTER_DIVISOR) / static_cast<float>(image_height);
                float box_width = (x_max - x_min) / static_cast<float>(image_width);
                float box_height = (y_max - y_min) / static_cast<float>(image_height);

                text_file << class_id << " " << x_center << " " << y_center << " " << box_width << " " << box_height
                          << "\n";
            }
        }
        converted_count++;
    }
    if (converted_count > 0)
    {
        LOG_INFO("XML -> YOLO conversion: {} .txt files created", converted_count);
    }
}

// Rewrites an OpenCV image (height, width, channel) into CHW, because the network tensor is NCHW.
auto hwc_to_chw(const cv::Mat& hwc, float* chw) -> void
{
    const int height = hwc.rows;
    const int width = hwc.cols;
    for (int row = 0; row < height; ++row)
    {
        const auto* pixel_row = hwc.ptr<cv::Vec3f>(row);
        for (int col = 0; col < width; ++col)
        {
            for (int channel = 0; channel < IMAGE_CHANNELS; ++channel)
            {
                chw[(channel * height * width) + (row * width) + col] = pixel_row[col][channel];
            }
        }
    }
}

// Scales and shifts the image around its center; pixels outside the frame stay black.
// The caller passes the same scale, dx, and dy into the labels so the box stays on the object.
auto apply_affine(cv::Mat& image, float scale, float dx, float dy) -> void
{
    const float half_width = static_cast<float>(image.cols) * 0.5F;
    // Both dx and dy use half the width, not the height, because the training image is square 448.
    const float translate_x = (0.5F * scale) + ((1.0F - scale + dx) * half_width) - 0.5F;
    const float translate_y = (0.5F * scale) + ((1.0F - scale + dy) * half_width) - 0.5F;
    cv::Mat affine = (cv::Mat_<float>(2, 3) << scale, 0.0F, translate_x, 0.0F, scale, translate_y);
    cv::warpAffine(image, image, affine, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.0, 0.0, 0.0));
}

// Multiplies saturation and value in HSV and clamps them to [0, 1].
// Only the train path calls this, so evaluation does not change colors.
auto apply_hsv_jitter(cv::Mat& image, float saturation_factor, float exposure_factor) -> void
{
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_RGB2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    channels[1] *= saturation_factor;
    channels[2] *= exposure_factor;
    cv::max(channels[1], 0.0, channels[1]);
    cv::min(channels[1], 1.0, channels[1]);
    cv::max(channels[2], 0.0, channels[2]);
    cv::min(channels[2], 1.0, channels[2]);
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, image, cv::COLOR_HSV2RGB);
}

// Writes YOLO-file boxes into the 7×7 grid: offset inside the cell, size, objectness, and the class one-hot from index 10.
// In train mode it undoes the image scale and shift; a third box in the same cell has no free slot left.
auto encode_targets(const std::string& label_path, bool is_train, float scale, float dx, float dy, int num_classes,
    std::vector<float>& target) -> void
{
    const int attributes = BOXES_PER_CELL * BOX_PARAMS + num_classes;
    target.assign(static_cast<std::size_t>(GRID_SIZE * GRID_SIZE * attributes), 0.0F);

    auto at = [&](int grid_y, int grid_x, int offset) -> float&
    {
        return target[static_cast<std::size_t>((grid_y * GRID_SIZE * attributes) + (grid_x * attributes) + offset)];
    };

    std::ifstream label_file(label_path);
    int class_id {};
    float x_center {};
    float y_center {};
    float box_width {};
    float box_height {};

    while (label_file >> class_id >> x_center >> y_center >> box_width >> box_height)
    {
        if (class_id < 0 || class_id >= num_classes)
        {
            continue;
        }

        if (is_train)
        {
            // Applies the inverse of the image affine at the box center, using the same scale, dx, and dy as warpAffine, so the box stays on the object.
            x_center = 0.5F * ((2.0F * x_center - 1.0F - dx) / scale + 1.0F);
            y_center = 0.5F * ((2.0F * y_center - 1.0F - dy) / scale + 1.0F);
            box_width = box_width / scale;
            box_height = box_height / scale;
        }

        if (x_center < 0.0F || x_center > 1.0F || y_center < 0.0F || y_center > 1.0F)
        {
            // A center outside [0, 1] does not enter the grid.
            continue;
        }

        box_width = std::clamp(box_width, 0.0F, 1.0F);
        box_height = std::clamp(box_height, 0.0F, 1.0F);

        int grid_x = std::clamp(static_cast<int>(x_center * static_cast<float>(GRID_SIZE)), 0, GRID_SIZE - 1);
        int grid_y = std::clamp(static_cast<int>(y_center * static_cast<float>(GRID_SIZE)), 0, GRID_SIZE - 1);

        for (int box_idx = 0; box_idx < BOXES_PER_CELL; ++box_idx)
        {
            int box_offset = box_idx * BOX_PARAMS;
            if (at(grid_y, grid_x, box_offset + 4) == 0.0F)
            {
                // x and y stored in the cell are offsets inside the cell, not coordinates of the whole image.
                at(grid_y, grid_x, box_offset + 0) = x_center * static_cast<float>(GRID_SIZE) - static_cast<float>(grid_x);
                at(grid_y, grid_x, box_offset + 1) = y_center * static_cast<float>(GRID_SIZE) - static_cast<float>(grid_y);
                at(grid_y, grid_x, box_offset + 2) = box_width;
                at(grid_y, grid_x, box_offset + 3) = box_height;
                // Objectness is slot 4. The class one-hot starts at index 10.
                at(grid_y, grid_x, box_offset + 4) = 1.0F;
                at(grid_y, grid_x, CLASS_OFFSET + class_id) = 1.0F;
                break;
            }
        }
    }
}
} // namespace

// Splits JPEG–label pairs into train, val, and test after a shuffle (default 0.7 / 0.15, the rest is test).
// Missing .txt files are created from VOC XML before the paths reach CustomDataLoader.
auto split_dataset(const std::string& voc_root, DataPaths& train_data, DataPaths& val_data, DataPaths& test_data,
    const std::vector<std::string>& class_names, float train_ratio, float val_ratio) -> void
{
    std::string jpeg_dir = voc_root + "/JPEGImages";
    std::string annot_dir = voc_root + "/Annotations";
    std::string label_dir = voc_root + "/labels";

    LOG_INFO("Looking for data in: {}", voc_root);

    if (!std::filesystem::exists(jpeg_dir) || !std::filesystem::exists(annot_dir))
    {
        LOG_ERROR("JPEGImages or Annotations not found!");
        return;
    }

    std::filesystem::create_directories(label_dir);

    bool conversion_needed = false;
    for (const auto& directory_entry : std::filesystem::directory_iterator(jpeg_dir))
    {
        if (directory_entry.path().extension() == ".jpg" && !std::filesystem::exists(label_dir + "/" + directory_entry.path().stem().string() + ".txt"))
        {
            conversion_needed = true;
            break;
        }
    }
    if (conversion_needed)
    {
        convert_voc_to_yolo(annot_dir, label_dir, jpeg_dir, class_names);
    }

    std::vector<std::pair<std::string, std::string>> all_pairs;
    for (const auto& directory_entry : std::filesystem::directory_iterator(jpeg_dir))
    {
        if (directory_entry.path().extension() == ".jpg")
        {
            std::string image_path = directory_entry.path().string();
            std::string base_name = directory_entry.path().stem().string();
            std::string label_path = label_dir + "/" + base_name + ".txt";
            if (std::filesystem::exists(label_path))
            {
                all_pairs.emplace_back(image_path, label_path);
            }
        }
    }

    if (all_pairs.empty())
    {
        LOG_ERROR("No paired images and labels found!");
        return;
    }

    // random_device seeds the shuffle so two runs do not share a file list.
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::shuffle(all_pairs.begin(), all_pairs.end(), generator);

    size_t total_elements = all_pairs.size();
    // The size_t cast truncates. Default ratios make train end floor(0.7 N) and val end that plus floor(0.15 N). The rest is test.
    auto train_end = static_cast<size_t>(static_cast<float>(total_elements) * train_ratio);
    auto val_end = train_end + static_cast<size_t>(static_cast<float>(total_elements) * val_ratio);

    for (size_t index = 0; index < total_elements; ++index)
    {
        if (index < train_end)
        {
            train_data.images.push_back(all_pairs[index].first);
            train_data.labels.push_back(all_pairs[index].second);
        }
        else if (index < val_end)
        {
            val_data.images.push_back(all_pairs[index].first);
            val_data.labels.push_back(all_pairs[index].second);
        }
        else
        {
            test_data.images.push_back(all_pairs[index].first);
            test_data.labels.push_back(all_pairs[index].second);
        }
    }

    LOG_INFO("Split complete. Total images: {}", total_elements);
    LOG_INFO("Train: {} | Val: {} | Test: {}", train_data.images.size(), val_data.images.size(),
        test_data.images.size());
}

// Checks the batch, the class list, and that the path counts match. reset sets the epoch order and the first prefetch.
CustomDataLoader::CustomDataLoader(const DataPaths& paths, int batch_size, bool is_train,
    const std::vector<std::string>& class_names)
    : paths_(paths)
    , batch_size_(batch_size)
    , is_train_(is_train)
    , num_classes_(static_cast<int>(class_names.size()))
    , img_size_(IMAGE_SIZE)
    , cursor_(0)
    , rng_(std::random_device {}())
{
    if (batch_size_ <= 0)
    {
        throw std::runtime_error("CustomDataLoader requires a positive batch size");
    }
    if (num_classes_ <= 0)
    {
        throw std::runtime_error("CustomDataLoader requires a non-empty class list");
    }
    if (paths_.images.size() != paths_.labels.size())
    {
        throw std::runtime_error("CustomDataLoader image and label path counts do not match");
    }
    reset();
}

// Finishes a pending prefetch before the paths and the generator used by the decode thread disappear.
CustomDataLoader::~CustomDataLoader()
{
    join_prefetch();
}

// Zeros the cursor and, in train mode, shuffles the indices. Prefetch of the first batch starts at once so the GPU does not wait on JPEG.
auto CustomDataLoader::reset() -> void
{
    join_prefetch();
    cursor_ = 0;
    order_.resize(paths_.images.size());
    std::iota(order_.begin(), order_.end(), 0);
    if (is_train_ && !order_.empty())
    {
        std::shuffle(order_.begin(), order_.end(), rng_);
    }
    launch_prefetch();
}

// Work remains when prefetch is already decoding a batch, or the cursor has not reached the end.
// The cursor alone is not enough: take_job advances it when prefetch starts.
auto CustomDataLoader::has_next() const -> bool
{
    return prefetch_.valid() || cursor_ < order_.size();
}

// Returns the number of paired images held by this loader.
auto CustomDataLoader::size() const -> std::size_t
{
    return paths_.images.size();
}

// Returns the requested batch size. The last batch of an epoch can be shorter; take_job computes that length.
auto CustomDataLoader::batch_size() const -> int
{
    return batch_size_;
}

// Loads a JPEG, resizes it to 448, and lays out CHW in [0, 1].
// Augmentation (scale, shift, HSV) runs only when is_train, so evaluation sees the image without distortion.
auto CustomDataLoader::load_sample(std::size_t sample_index, std::vector<float>& image_chw, std::vector<float>& target,
    std::mt19937& rng) const -> void
{
    const std::size_t image_elems = static_cast<std::size_t>(IMAGE_CHANNELS * img_size_ * img_size_);
    image_chw.assign(image_elems, 0.0F);

    float scale = 1.0F;
    float dx = 0.0F;
    float dy = 0.0F;

    cv::Mat image = cv::imread(paths_.images[sample_index]);
    if (image.empty())
    {
        // A failed read leaves a zero CHW buffer and an unscaled label so the batch keeps a fixed shape.
        encode_targets(paths_.labels[sample_index], false, 1.0F, 0.0F, 0.0F, num_classes_, target);
        return;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(img_size_, img_size_));
    // Pixels are divided by 255.
    image.convertTo(image, CV_32FC3, 1.0F / NORMALIZATION_FACTOR);

    if (is_train_)
    {
        std::uniform_real_distribution<float> scale_dist(0.8F, 1.2F);
        std::uniform_real_distribution<float> shift_dist(-0.2F, 0.2F);
        std::uniform_real_distribution<float> jitter_dist(0.66F, 1.5F);
        scale = scale_dist(rng);
        dx = shift_dist(rng);
        dy = shift_dist(rng);
        apply_affine(image, scale, dx, dy);
        // HSV jitter only when is_train.
        apply_hsv_jitter(image, jitter_dist(rng), jitter_dist(rng));
    }

    if (!image.isContinuous())
    {
        image = image.clone();
    }
    hwc_to_chw(image, image_chw.data());
    encode_targets(paths_.labels[sample_index], is_train_, scale, dx, dy, num_classes_, target);
}

// Takes the next index range and draws one RNG seed per sample, so decode threads do not share one generator.
auto CustomDataLoader::take_job() -> std::pair<std::vector<std::size_t>, std::vector<std::uint32_t>>
{
    std::vector<std::size_t> sample_indices;
    std::vector<std::uint32_t> rng_seeds;
    if (cursor_ >= order_.size())
    {
        return { sample_indices, rng_seeds };
    }
    const std::size_t remaining = order_.size() - cursor_;
    const int this_batch = static_cast<int>(std::min(remaining, static_cast<std::size_t>(batch_size_)));
    sample_indices.resize(static_cast<std::size_t>(this_batch));
    rng_seeds.resize(static_cast<std::size_t>(this_batch));
    for (int batch_idx = 0; batch_idx < this_batch; ++batch_idx)
    {
        sample_indices[static_cast<std::size_t>(batch_idx)] = order_[cursor_++];
        rng_seeds[static_cast<std::size_t>(batch_idx)] = rng_();
    }
    return { std::move(sample_indices), std::move(rng_seeds) };
}

// Decodes the batch samples in parallel into host buffers (CHW image and YOLO grid).
// Parallel JPEG reads shorten the time the GPU would otherwise spend waiting on data.
auto CustomDataLoader::decode_job(std::vector<std::size_t> sample_indices, std::vector<std::uint32_t> rng_seeds) const
    -> HostBatch
{
    HostBatch host;
    host.n = static_cast<int>(sample_indices.size());
    host.attributes = BOXES_PER_CELL * BOX_PARAMS + num_classes_;
    if (host.n == 0)
    {
        return host;
    }

    const std::size_t image_elems = static_cast<std::size_t>(IMAGE_CHANNELS * img_size_ * img_size_);
    const std::size_t target_elems = static_cast<std::size_t>(GRID_SIZE * GRID_SIZE * host.attributes);
    host.images.assign(static_cast<std::size_t>(host.n) * image_elems, 0.0F);
    host.targets.assign(static_cast<std::size_t>(host.n) * target_elems, 0.0F);

    dl::parallel_for(host.n,
        [this, &sample_indices, &rng_seeds, &host, image_elems, target_elems](int batch_idx)
        {
            std::mt19937 local_rng(rng_seeds[static_cast<std::size_t>(batch_idx)]);
            std::vector<float> sample_image;
            std::vector<float> sample_target;
            load_sample(sample_indices[static_cast<std::size_t>(batch_idx)], sample_image, sample_target, local_rng);

            const auto image_offset = static_cast<std::ptrdiff_t>(batch_idx) * static_cast<std::ptrdiff_t>(image_elems);
            const auto target_offset = static_cast<std::ptrdiff_t>(batch_idx) * static_cast<std::ptrdiff_t>(target_elems);
            if (sample_image.size() == image_elems)
            {
                std::copy(sample_image.begin(), sample_image.end(), host.images.begin() + image_offset);
            }
            if (sample_target.size() == target_elems)
            {
                std::copy(sample_target.begin(), sample_target.end(), host.targets.begin() + target_offset);
            }
        });
    return host;
}

// Uploads images and the grid to the GPU through from_host, on the stream given by the caller.
auto CustomDataLoader::upload_host_batch(HostBatch host, cudaStream_t stream) const -> Batch
{
    return Batch { dl::Tensor::from_host({ host.n, IMAGE_CHANNELS, img_size_, img_size_ }, host.images, dl::Device::GPU,
                       stream, dl::Dtype::Float32),
        dl::Tensor::from_host({ host.n, GRID_SIZE, GRID_SIZE, host.attributes }, host.targets, dl::Device::GPU, stream,
            dl::Dtype::Float32) };
}

// Starts decoding the next batch through std::async, so it does not wait for the current GPU step to finish.
auto CustomDataLoader::launch_prefetch() -> void
{
    auto job = take_job();
    if (job.first.empty())
    {
        return;
    }
    // Prefetch decodes the next batch so the GPU does not wait on JPEG.
    prefetch_ = std::async(std::launch::async,
        [this, indices = std::move(job.first), seeds = std::move(job.second)]()
        { return decode_job(std::move(indices), std::move(seeds)); });
}

// Waits for the prefetch thread and drops the result, so reset and the destructor do not leave a decode running.
auto CustomDataLoader::join_prefetch() -> void
{
    if (!prefetch_.valid())
    {
        return;
    }
    try
    {
        prefetch_.wait();
    }
    catch (...)
    {
    }
    prefetch_ = {};
}

// Returns the ready batch and immediately starts decoding the next one, before from_host returns.
// CPU prefetch means the GPU does not wait on JPEG while it computes the current step.
auto CustomDataLoader::get_batch(cudaStream_t stream) -> Batch
{
    const dl::NvtxRange nvtx_range("DataLoader_GetBatch");
    if (!prefetch_.valid())
    {
        launch_prefetch();
    }
    if (!prefetch_.valid())
    {
        throw std::runtime_error("CustomDataLoader::get_batch called with no remaining samples");
    }

    HostBatch host = prefetch_.get();
    prefetch_ = {};
    launch_prefetch();
    if (host.n <= 0)
    {
        throw std::runtime_error("CustomDataLoader decoded an empty batch");
    }
    return upload_host_batch(std::move(host), stream);
}
