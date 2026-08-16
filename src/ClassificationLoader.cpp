#include "DeepLearnLib/ClassificationLoader.hpp"

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

auto is_image_file(const fs::path& path) -> bool
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp";
}

auto hwc_to_chw(const cv::Mat& image, float* destination) -> void
{
    const int height = image.rows;
    const int width = image.cols;
    const std::size_t plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    for (int row = 0; row < height; ++row)
    {
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

ClassificationLoader::ClassificationLoader(std::string dataset_root, std::string split, int batch_size, int image_size,
    bool shuffle)
    : batch_size_(batch_size)
    , image_size_(image_size)
    , shuffle_(shuffle)
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
    if (!fs::is_directory(split_path))
    {
        split_path = root;
    }
    if (!fs::is_directory(split_path))
    {
        throw std::runtime_error("ClassificationLoader could not find split directory: " + split_path.string());
    }
    split_root_ = split_path.string();

    for (const auto& entry : fs::directory_iterator(split_path))
    {
        if (entry.is_directory())
        {
            class_names_.push_back(entry.path().filename().string());
        }
    }
    std::sort(class_names_.begin(), class_names_.end());
    if (class_names_.empty())
    {
        throw std::runtime_error("ClassificationLoader found no class folders in " + split_root_);
    }

    for (int class_id = 0; class_id < static_cast<int>(class_names_.size()); ++class_id)
    {
        const fs::path class_dir = split_path / class_names_[static_cast<std::size_t>(class_id)];
        for (const auto& file : fs::directory_iterator(class_dir))
        {
            if (file.is_regular_file() && is_image_file(file.path()))
            {
                samples_.emplace_back(file.path().string(), class_id);
            }
        }
    }
    if (samples_.empty())
    {
        throw std::runtime_error("ClassificationLoader found no images in " + split_root_);
    }
    reset();
}

auto ClassificationLoader::reset() -> void
{
    cursor_ = 0;
    order_.resize(samples_.size());
    std::iota(order_.begin(), order_.end(), 0);
    if (shuffle_ && !order_.empty())
    {
        std::shuffle(order_.begin(), order_.end(), rng_);
    }
}

auto ClassificationLoader::has_next() const -> bool
{
    return cursor_ < order_.size();
}

auto ClassificationLoader::size() const -> std::size_t
{
    return samples_.size();
}

auto ClassificationLoader::batch_size() const -> int
{
    return batch_size_;
}

auto ClassificationLoader::num_classes() const -> int
{
    return static_cast<int>(class_names_.size());
}

auto ClassificationLoader::image_size() const -> int
{
    return image_size_;
}

auto ClassificationLoader::class_names() const -> const std::vector<std::string>&
{
    return class_names_;
}

auto ClassificationLoader::load_sample(std::size_t sample_index, std::vector<float>& image_chw) -> int
{
    const std::size_t image_elems = static_cast<std::size_t>(kChannels * image_size_ * image_size_);
    image_chw.assign(image_elems, 0.0F);

    cv::Mat image = cv::imread(samples_[sample_index].first);
    if (image.empty())
    {
        return samples_[sample_index].second;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(image_size_, image_size_));
    image.convertTo(image, CV_32FC3, 1.0F / kNorm);
    if (!image.isContinuous())
    {
        image = image.clone();
    }
    hwc_to_chw(image, image_chw.data());
    return samples_[sample_index].second;
}

auto ClassificationLoader::get_batch() -> Batch
{
    if (!has_next())
    {
        throw std::runtime_error("ClassificationLoader::get_batch called with no remaining samples");
    }

    const std::size_t remaining = order_.size() - cursor_;
    const int this_batch = static_cast<int>(std::min(remaining, static_cast<std::size_t>(batch_size_)));
    const int classes = num_classes();
    const std::size_t image_elems = static_cast<std::size_t>(kChannels * image_size_ * image_size_);
    const std::size_t target_elems = static_cast<std::size_t>(classes);

    std::vector<float> images_host(static_cast<std::size_t>(this_batch) * image_elems, 0.0F);
    std::vector<float> targets_host(static_cast<std::size_t>(this_batch) * target_elems, 0.0F);
    std::vector<float> sample_image;
    for (int batch_idx = 0; batch_idx < this_batch; ++batch_idx)
    {
        const std::size_t sample_index = order_[cursor_++];
        const int class_id = load_sample(sample_index, sample_image);
        std::copy(sample_image.begin(), sample_image.end(),
            images_host.begin() + (static_cast<std::ptrdiff_t>(batch_idx) * static_cast<std::ptrdiff_t>(image_elems)));
        const int clamped = std::clamp(class_id, 0, classes - 1);
        targets_host[(static_cast<std::size_t>(batch_idx) * target_elems) + static_cast<std::size_t>(clamped)] = 1.0F;
    }

    return Batch { dl::Tensor::from_host({ this_batch, kChannels, image_size_, image_size_ }, images_host,
                       dl::Device::GPU, 0, dl::compute_dtype()),
        dl::Tensor::from_host({ this_batch, classes }, targets_host, dl::Device::GPU, 0, dl::compute_dtype()) };
}
