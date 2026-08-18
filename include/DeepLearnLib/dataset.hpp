#pragma once

#include "DeepLearnLib/Tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <random>
#include <string>
#include <utility>
#include <vector>

extern const std::vector<std::string> VOC_CLASSES_DEFAULT;

/**
 * Parallel lists of image paths and matching annotation paths.
 */
struct DataPaths
{
    std::vector<std::string> images;
    std::vector<std::string> labels;
};

void split_dataset(const std::string& voc_root, DataPaths& train, DataPaths& val, DataPaths& test,
    const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT, float train_ratio = 0.7F,
    float val_ratio = 0.15F);

inline void splitDataset(const std::string& voc_root, DataPaths& train, DataPaths& val, DataPaths& test,
    const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT, float train_ratio = 0.7F,
    float val_ratio = 0.15F)
{
    split_dataset(voc_root, train, val, test, class_names, train_ratio, val_ratio);
}

/** Target tensor layout written by CustomDataLoader. */
enum class DetectionLabelLayout
{
    /** Paper YOLOv1: `[S, S, B*5 + C]` with B=2. */
    PaperYolov1,
    /** Darknet detection truth: `[S, S, 1 + 4 + C]` (`is_obj`, classes, x, y, w, h). */
    DarknetYolov1
};

/**
 * One GPU-resident training/evaluation batch.
 *
 * Detection loaders use images [N, 3, 448, 448] and YOLO grids as targets.
 * ClassificationLoader uses images [N, 3, H, W] and one-hot targets [N, C].
 * PackedImageLoader uses packed `DLIMG001` binaries (MNIST) with any channel count.
 */
struct Batch
{
    dl::Tensor images;
    dl::Tensor targets;
};

/**
 * Sequential/shuffled mini-batch loader (OpenCV decode + dl::Tensor upload).
 *
 * Training applies affine scale/translation and HSV jitter on the CPU, then
 * uploads CHW images and YOLO targets with from_host. Decode uses a bounded
 * thread pool and the next host batch is prefetched during GPU compute.
 */
class CustomDataLoader
{
public:
    CustomDataLoader(const DataPaths& paths, int batch_size, bool is_train,
        const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT,
        DetectionLabelLayout label_layout = DetectionLabelLayout::PaperYolov1);
    ~CustomDataLoader();

    CustomDataLoader(const CustomDataLoader&) = delete;
    auto operator=(const CustomDataLoader&) -> CustomDataLoader& = delete;
    CustomDataLoader(CustomDataLoader&&) = delete;
    auto operator=(CustomDataLoader&&) -> CustomDataLoader& = delete;

    auto reset() -> void;
    [[nodiscard]] auto has_next() const -> bool;
    auto get_batch(cudaStream_t stream = 0) -> Batch;
    [[nodiscard]] auto size() const -> std::size_t;
    [[nodiscard]] auto batch_size() const -> int;

private:
    struct HostBatch
    {
        int n { 0 };
        int attributes { 0 };
        std::vector<float> images;
        std::vector<float> targets;
    };

    DataPaths paths_;
    int batch_size_;
    bool is_train_;
    int num_classes_;
    int img_size_;
    DetectionLabelLayout label_layout_;
    std::size_t cursor_;
    std::vector<std::size_t> order_;
    std::mt19937 rng_;
    std::future<HostBatch> prefetch_;

    auto load_sample(std::size_t sample_index, std::vector<float>& image_chw, std::vector<float>& target,
        std::mt19937& rng) const -> void;
    auto take_job() -> std::pair<std::vector<std::size_t>, std::vector<std::uint32_t>>;
    [[nodiscard]] auto decode_job(std::vector<std::size_t> sample_indices, std::vector<std::uint32_t> rng_seeds) const
        -> HostBatch;
    auto upload_host_batch(HostBatch host, cudaStream_t stream) const -> Batch;
    auto launch_prefetch() -> void;
    auto join_prefetch() -> void;
};
