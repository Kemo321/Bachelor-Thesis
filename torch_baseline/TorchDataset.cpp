#include "TorchDataset.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace
{
constexpr int GRID_SIZE = 7;
constexpr int BOXES_PER_CELL = 2;
constexpr int BOX_PARAMS = 5;
constexpr int CLASS_OFFSET = 10;
constexpr float NORMALIZATION_FACTOR = 255.0F;
constexpr int IMAGE_CHANNELS = 3;
} // namespace

VOCYoloDataset::VOCYoloDataset(const DataPaths& paths_param, bool is_train, const std::vector<std::string>& class_names)
    : paths(paths_param), is_train_(is_train), num_classes_(static_cast<int>(class_names.size()))
{
}

auto VOCYoloDataset::get(size_t index) -> torch::data::Example<>
{
    int bbox_output_size = BOXES_PER_CELL * BOX_PARAMS + num_classes_;

    int image_width{};
    int image_height{};
    int image_channels{};
    unsigned char* image_data =
        stbi_load(paths.images[index].c_str(), &image_width, &image_height, &image_channels, IMAGE_CHANNELS);
    if (image_data == nullptr)
    {
        return { torch::zeros({ IMAGE_CHANNELS, img_size, img_size }),
                 torch::zeros({ GRID_SIZE, GRID_SIZE, bbox_output_size }) };
    }

    auto image_tensor = torch::from_blob(image_data, { image_height, image_width, IMAGE_CHANNELS }, torch::kUInt8).clone();
    stbi_image_free(image_data);

    image_tensor = image_tensor.to(torch::kFloat32).div_(NORMALIZATION_FACTOR).permute({ 2, 0, 1 }).contiguous();

    image_tensor = torch::nn::functional::interpolate(
                       image_tensor.unsqueeze(0),
                       torch::nn::functional::InterpolateFuncOptions()
                           .size(std::vector<int64_t>{ img_size, img_size })
                           .mode(torch::kBilinear)
                           .align_corners(false))
                       .squeeze(0);

    float scale = 1.0F;
    float dx = 0.0F;
    float dy = 0.0F;

    if (is_train_)
    {
        scale = 0.8F + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 0.4F));
        dx = -0.2F + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 0.4F));
        dy = -0.2F + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 0.4F));

        torch::Tensor theta = torch::tensor({ { { scale, 0.0F, dx }, { 0.0F, scale, dy } } }, torch::kFloat32);
        auto grid = torch::nn::functional::affine_grid(theta, { 1, IMAGE_CHANNELS, img_size, img_size }, false);
        image_tensor =
            torch::nn::functional::grid_sample(image_tensor.unsqueeze(0), grid,
                                               torch::nn::functional::GridSampleFuncOptions()
                                                   .mode(torch::kBilinear)
                                                   .padding_mode(torch::kZeros)
                                                   .align_corners(false))
                .squeeze(0)
                .contiguous();

        float saturation_factor = 0.66F + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 0.84F));
        float exposure_factor = 0.66F + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 0.84F));

        image_tensor = torch::clamp(image_tensor * exposure_factor, 0.0F, 1.0F);
        auto grayscale = image_tensor.mean(0, true).expand_as(image_tensor).contiguous();
        image_tensor = torch::clamp(grayscale + saturation_factor * (image_tensor - grayscale), 0.0F, 1.0F);
    }

    torch::Tensor target_tensor = torch::zeros({ GRID_SIZE, GRID_SIZE, bbox_output_size });

    std::ifstream label_file(paths.labels[index]);
    int class_id{};
    float x_center{};
    float y_center{};
    float box_width{};
    float box_height{};

    while (label_file >> class_id >> x_center >> y_center >> box_width >> box_height)
    {
        if (is_train_)
        {
            x_center = 0.5F * ((2.0F * x_center - 1.0F - dx) / scale + 1.0F);
            y_center = 0.5F * ((2.0F * y_center - 1.0F - dy) / scale + 1.0F);
            box_width = box_width / scale;
            box_height = box_height / scale;
        }

        if (x_center < 0.0F || x_center > 1.0F || y_center < 0.0F || y_center > 1.0F)
        {
            continue;
        }

        box_width = std::clamp(box_width, 0.0F, 1.0F);
        box_height = std::clamp(box_height, 0.0F, 1.0F);

        int grid_x = std::clamp(static_cast<int>(x_center * static_cast<float>(GRID_SIZE)), 0, GRID_SIZE - 1);
        int grid_y = std::clamp(static_cast<int>(y_center * static_cast<float>(GRID_SIZE)), 0, GRID_SIZE - 1);

        for (int box_idx = 0; box_idx < BOXES_PER_CELL; ++box_idx)
        {
            int offset_val = box_idx * BOX_PARAMS;
            if (target_tensor[grid_y][grid_x][offset_val + 4].item<float>() == 0.0F)
            {
                target_tensor[grid_y][grid_x][offset_val + 0] =
                    x_center * static_cast<float>(GRID_SIZE) - static_cast<float>(grid_x);
                target_tensor[grid_y][grid_x][offset_val + 1] =
                    y_center * static_cast<float>(GRID_SIZE) - static_cast<float>(grid_y);
                target_tensor[grid_y][grid_x][offset_val + 2] = box_width;
                target_tensor[grid_y][grid_x][offset_val + 3] = box_height;
                target_tensor[grid_y][grid_x][offset_val + 4] = 1.0F;
                target_tensor[grid_y][grid_x][CLASS_OFFSET + class_id] = 1.0F;
                break;
            }
        }
    }
    return { image_tensor, target_tensor };
}

auto VOCYoloDataset::size() const -> torch::optional<size_t>
{
    return paths.images.size();
}
