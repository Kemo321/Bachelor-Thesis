#include "DeepLearnLib/ClassificationLoader.hpp"
#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Nvtx.hpp"
#include "DeepLearnLib/ParallelFor.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace
{

constexpr int kChannels = 3;
constexpr float kNorm = 255.0F;

// Recognizes jpg, jpeg, png, and bmp regardless of the extension's letter case.
auto is_image_file(const fs::path& path) -> bool
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char character)
        { return static_cast<char>(std::tolower(character)); });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp";
}

// Splits interleaved OpenCV RGB into three CHW planes, because the network reads NCHW.
auto hwc_to_chw(const cv::Mat& image, float* destination) -> void
{
    const int height = image.rows;
    const int width = image.cols;
    const std::size_t plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    for (int row = 0; row < height; ++row)
    {
        // A CV_32FC3 row: three floats per pixel, channels interleaved.
        const auto* pixel = image.ptr<float>(row);
        for (int col = 0; col < width; ++col)
        {
            const std::size_t spatial = (static_cast<std::size_t>(row) * static_cast<std::size_t>(width))
                + static_cast<std::size_t>(col);
            destination[spatial] = pixel[0];
            destination[plane + spatial] = pixel[1];
            destination[(2 * plane) + spatial] = pixel[2];
            pixel += 3;
        }
    }
}

} // namespace

// Collects images from class directories and fixes the order that later sets the one-hot position.
// A supplied class_names list keeps the same class index when a split is missing a folder.
ClassificationLoader::ClassificationLoader(std::string dataset_root, std::string split, int batch_size, int image_size,
    bool shuffle, std::vector<std::string> class_names)
    : batch_size_(batch_size)
    , image_size_(image_size)
    , shuffle_(shuffle)
    , class_names_(std::move(class_names))
    , rng_(std::random_device {}())
{
    if (batch_size_ <= 0)
    {
        throw std::runtime_error("ClassificationLoader requires a positive batch size");
    }
    if (image_size_ <= 0)
    {
        throw std::runtime_error("ClassificationLoader requires a positive image size");
    }

    fs::path root(dataset_root);
    fs::path split_path = root / split;
    // When root/split is not a directory, the images live directly in root.
    if (!fs::is_directory(split_path))
    {
        split_path = root;
    }
    if (!fs::is_directory(split_path))
    {
        throw std::runtime_error("ClassificationLoader could not find split directory: " + split_path.string());
    }
    split_root_ = split_path.string();

    if (class_names_.empty())
    {
        for (const auto& entry : fs::directory_iterator(split_path))
        {
            if (entry.is_directory())
            {
                class_names_.push_back(entry.path().filename().string());
            }
        }
        std::sort(class_names_.begin(), class_names_.end());
    }
    if (class_names_.empty())
    {
        throw std::runtime_error("ClassificationLoader found no class folders in " + split_root_);
    }

    for (int class_id = 0; class_id < static_cast<int>(class_names_.size()); ++class_id)
    {
        const fs::path class_dir = split_path / class_names_[static_cast<std::size_t>(class_id)];
        if (!fs::is_directory(class_dir))
        {
            continue;
        }
        for (const auto& file : fs::directory_iterator(class_dir))
        {
            if (file.is_regular_file() && is_image_file(file.path()))
            {
                // The class-directory index is the one-hot class.
                samples_.emplace_back(file.path().string(), class_id);
            }
        }
    }
    if (samples_.empty())
    {
        throw std::runtime_error("ClassificationLoader found no images in " + split_root_);
    }
    LOG_DEBUG("ClassificationLoader {} classes={} images={} batch={} image_size={} shuffle={} decode_workers={}",
        split_root_, class_names_.size(), samples_.size(), batch_size_, image_size_, shuffle_,
        dl::parallel_worker_count(batch_size_));
    reset();
}

// Finishes prefetch before the samples used by the decode thread disappear.
ClassificationLoader::~ClassificationLoader()
{
    join_prefetch();
}

// Restarts the epoch and, when shuffle is set, shuffles the indices, then decodes the first batch in the background.
auto ClassificationLoader::reset() -> void
{
    join_prefetch();
    cursor_ = 0;
    order_.resize(samples_.size());
    std::iota(order_.begin(), order_.end(), 0);
    if (shuffle_ && !order_.empty())
    {
        std::shuffle(order_.begin(), order_.end(), rng_);
    }
    launch_prefetch();
}

// A batch remains in prefetch, or unread indices remain.
// The cursor moves when prefetch starts, so it alone does not describe the last batch.
auto ClassificationLoader::has_next() const -> bool
{
    return prefetch_.valid() || cursor_ < order_.size();
}

// Returns the number of images collected in this split.
auto ClassificationLoader::size() const -> std::size_t
{
    return samples_.size();
}

// Returns the requested batch size. The last batch of an epoch can be shorter.
auto ClassificationLoader::batch_size() const -> int
{
    return batch_size_;
}

// Returns the class count, which is the width of the one-hot vector.
auto ClassificationLoader::num_classes() const -> int
{
    return static_cast<int>(class_names_.size());
}

// Returns the side of the square that images are scaled to before upload to the GPU.
auto ClassificationLoader::image_size() const -> int
{
    return image_size_;
}

// Returns class names in the order that sets the one-hot index.
auto ClassificationLoader::class_names() const -> const std::vector<std::string>&
{
    return class_names_;
}

// Loads an image, converts BGR to RGB and, when a side differs, scales to image_size, then divides by 255.
// Returns the class id from the directory, including when the file read fails and the image stays zeros.
auto ClassificationLoader::load_sample(std::size_t sample_index, std::vector<float>& image_chw) const -> int
{
    const std::size_t image_elems = static_cast<std::size_t>(kChannels * image_size_ * image_size_);
    image_chw.assign(image_elems, 0.0F);

    cv::Mat image = cv::imread(samples_[sample_index].first, cv::IMREAD_COLOR);
    if (image.empty())
    {
        return samples_[sample_index].second;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    if (image.cols != image_size_ || image.rows != image_size_)
    {
        cv::resize(image, image, cv::Size(image_size_, image_size_), 0.0, 0.0, cv::INTER_LINEAR);
    }
    // Pixels are divided by 255.
    image.convertTo(image, CV_32FC3, 1.0F / kNorm);
    if (!image.isContinuous())
    {
        image = image.clone();
    }
    hwc_to_chw(image, image_chw.data());
    return samples_[sample_index].second;
}

// Takes the next index range from order_ and advances the cursor by that batch.
auto ClassificationLoader::take_indices() -> std::vector<std::size_t>
{
    std::vector<std::size_t> indices;
    if (cursor_ >= order_.size())
    {
        return indices;
    }
    const std::size_t remaining = order_.size() - cursor_;
    const int this_batch = static_cast<int>(std::min(remaining, static_cast<std::size_t>(batch_size_)));
    indices.reserve(static_cast<std::size_t>(this_batch));
    for (int batch_idx = 0; batch_idx < this_batch; ++batch_idx)
    {
        indices.push_back(order_[cursor_++]);
    }
    return indices;
}

// Decodes images in parallel and places a single 1 in the class vector (one-hot [N, C]).
auto ClassificationLoader::decode_indices(const std::vector<std::size_t>& indices) const -> HostBatch
{
    HostBatch host;
    host.n = static_cast<int>(indices.size());
    if (host.n == 0)
    {
        return host;
    }

    const int classes = num_classes();
    const std::size_t image_elems = static_cast<std::size_t>(kChannels * image_size_ * image_size_);
    const std::size_t target_elems = static_cast<std::size_t>(classes);
    host.images.assign(static_cast<std::size_t>(host.n) * image_elems, 0.0F);
    host.targets.assign(static_cast<std::size_t>(host.n) * target_elems, 0.0F);

    dl::parallel_for(host.n,
        [this, &indices, &host, image_elems, target_elems, classes](int batch_idx)
        {
            std::vector<float> sample_image;
            const int class_id = load_sample(indices[static_cast<std::size_t>(batch_idx)], sample_image);
            const auto image_offset = static_cast<std::ptrdiff_t>(batch_idx) * static_cast<std::ptrdiff_t>(image_elems);
            if (sample_image.size() == image_elems)
            {
                std::copy(sample_image.begin(), sample_image.end(), host.images.begin() + image_offset);
            }
            const int clamped = std::clamp(class_id, 0, classes - 1);
            // A class id outside 0…C-1 does not write past the one-hot row.
            host.targets[(static_cast<std::size_t>(batch_idx) * target_elems) + static_cast<std::size_t>(clamped)] = 1.0F;
        });
    return host;
}

// Uploads NCHW images and the one-hot to the GPU through from_host, on the caller's stream.
auto ClassificationLoader::upload_host_batch(HostBatch host, cudaStream_t stream) const -> Batch
{
    return Batch { dl::Tensor::from_host({ host.n, kChannels, image_size_, image_size_ }, host.images, dl::Device::GPU,
                       stream, dl::Dtype::Float32),
        dl::Tensor::from_host({ host.n, num_classes() }, host.targets, dl::Device::GPU, stream, dl::Dtype::Float32) };
}

// Starts decoding the next batch in the background so the GPU does not wait on JPEG.
auto ClassificationLoader::launch_prefetch() -> void
{
    auto indices = take_indices();
    if (indices.empty())
    {
        return;
    }
    // Prefetch decodes the next batch so the GPU does not wait on JPEG.
    prefetch_ = std::async(std::launch::async, [this, indices = std::move(indices)]()
        { return decode_indices(indices); });
}

// Waits for prefetch and drops the result before reset or destruction of the loader.
auto ClassificationLoader::join_prefetch() -> void
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

// Returns the decoded batch and immediately starts the next one, before from_host returns.
// JPEG prefetch on the CPU overlaps the current step on the GPU.
auto ClassificationLoader::get_batch(cudaStream_t stream) -> Batch
{
    const dl::NvtxRange nvtx_range("ClassificationLoader_GetBatch");
    if (!prefetch_.valid())
    {
        launch_prefetch();
    }
    if (!prefetch_.valid())
    {
        throw std::runtime_error("ClassificationLoader::get_batch called with no remaining samples");
    }

    HostBatch host = prefetch_.get();
    prefetch_ = {};
    launch_prefetch();
    if (host.n <= 0)
    {
        throw std::runtime_error("ClassificationLoader decoded an empty batch");
    }
    return upload_host_batch(std::move(host), stream);
}
