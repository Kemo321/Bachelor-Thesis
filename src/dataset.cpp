#include "DeepLearnLib/dataset.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pugixml.hpp>
#include <random>
#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace
{
constexpr int GRID_SIZE = 7;
constexpr int BBOX_OUTPUT_SIZE = 30;
constexpr int BOXES_PER_CELL = 2;
constexpr int BOX_PARAMS = 5;
constexpr int CLASS_OFFSET = 10;
constexpr float NORMALIZATION_FACTOR = 255.0F;
constexpr float CENTER_DIVISOR = 2.0F;
constexpr int IMAGE_CHANNELS = 3;

auto voc_class_to_id(const std::string& class_name) -> int
{
    static const std::unordered_map<std::string, int> voc_classes = {
        { "aeroplane", 0 }, { "bicycle", 1 }, { "bird", 2 }, { "boat", 3 }, { "bottle", 4 },
        { "bus", 5 }, { "car", 6 }, { "cat", 7 }, { "chair", 8 }, { "cow", 9 },
        { "diningtable", 10 }, { "dog", 11 }, { "horse", 12 }, { "motorbike", 13 },
        { "person", 14 }, { "pottedplant", 15 }, { "sheep", 16 }, { "sofa", 17 },
        { "train", 18 }, { "tvmonitor", 19 }
    };
    auto iterator = voc_classes.find(class_name);
    return (iterator != voc_classes.end()) ? iterator->second : -1;
}

auto convert_voc_to_yolo(const std::string& annot_dir, const std::string& label_dir, const std::string& jpeg_dir) -> void
{
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
        for (auto object_node = root_node.child("object"); object_node != nullptr; object_node = object_node.next_sibling("object"))
        {
            std::string class_name = object_node.child("name").text().as_string();
            auto bound_box = object_node.child("bndbox");
            if (bound_box.empty())
            {
                continue;
            }

            int class_id = voc_class_to_id(class_name);
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

                text_file << class_id << " " << x_center << " " << y_center << " " << box_width << " " << box_height << "\n";
            }
        }
        converted_count++;
    }
    if (converted_count > 0)
    {
        std::cout << "[INFO] Konwersja XML -> YOLO: " << converted_count << " plikow .txt utworzonych\n";
    }
}
}

auto split_dataset(const std::string& voc_root, DataPaths& train_data, DataPaths& val_data, DataPaths& test_data, float train_ratio, float val_ratio) -> void
{
    std::string jpeg_dir = voc_root + "/JPEGImages";
    std::string annot_dir = voc_root + "/Annotations";
    std::string label_dir = voc_root + "/labels";

    std::cout << "[INFO] Szukam danych VOC w: " << voc_root << "\n";

    if (!std::filesystem::exists(jpeg_dir) || !std::filesystem::exists(annot_dir))
    {
        std::cerr << "[BLAD] Nie znaleziono JPEGImages lub Annotations!\n";
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
        convert_voc_to_yolo(annot_dir, label_dir, jpeg_dir);
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
        std::cerr << "[BLAD] Brak sparowanych obrazow i etykiet!\n";
        return;
    }

    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::shuffle(all_pairs.begin(), all_pairs.end(), generator);

    size_t total_elements = all_pairs.size();
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

    std::cout << "[INFO] Podzial zakonczony! Razem: " << total_elements << " obrazow\n";
    std::cout << "   Train: " << train_data.images.size() << " | Val: " << val_data.images.size()
              << " | Test: " << test_data.images.size() << "\n\n";
}

VOCYoloDataset::VOCYoloDataset(const DataPaths& paths_param, bool is_train)
    : paths(paths_param), is_train_(is_train)
{
}

auto VOCYoloDataset::get(size_t index) -> torch::data::Example<>
{
    int image_width{};
    int image_height{};
    int image_channels{};
    unsigned char* image_data = stbi_load(paths.images[index].c_str(), &image_width, &image_height, &image_channels, IMAGE_CHANNELS);
    if (image_data == nullptr)
    {
        return { torch::zeros({ IMAGE_CHANNELS, img_size, img_size }), torch::zeros({ GRID_SIZE, GRID_SIZE, BBOX_OUTPUT_SIZE }) };
    }

    auto image_tensor = torch::from_blob(image_data, { image_height, image_width, IMAGE_CHANNELS }, torch::kUInt8).clone();
    stbi_image_free(image_data);

    image_tensor = image_tensor.to(torch::kFloat32).div_(NORMALIZATION_FACTOR).permute({ 2, 0, 1 });
    
    image_tensor = torch::nn::functional::interpolate(
        image_tensor.unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{ img_size, img_size })
            .mode(torch::kBilinear)
            .align_corners(false)).squeeze(0);

    float scale = 1.0F;
    float dx = 0.0F;
    float dy = 0.0F;

    if (is_train_)
    {
        scale = 0.8F + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 0.4F));
        dx = -0.2F + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 0.4F));
        dy = -0.2F + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 0.4F));

        torch::Tensor theta = torch::tensor({{{scale, 0.0F, dx}, {0.0F, scale, dy}}}, torch::kFloat32);
        auto grid = torch::nn::functional::affine_grid(theta, {1, IMAGE_CHANNELS, img_size, img_size}, false);
        image_tensor = torch::nn::functional::grid_sample(image_tensor.unsqueeze(0), grid, 
            torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false)).squeeze(0);

        float saturation_factor = 0.66F + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 0.84F));
        float exposure_factor = 0.66F + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 0.84F));

        image_tensor = torch::clamp(image_tensor * exposure_factor, 0.0F, 1.0F);
        auto grayscale = image_tensor.mean(0, true).expand_as(image_tensor);
        image_tensor = torch::clamp(grayscale + saturation_factor * (image_tensor - grayscale), 0.0F, 1.0F);
    }

    torch::Tensor target_tensor = torch::zeros({ GRID_SIZE, GRID_SIZE, BBOX_OUTPUT_SIZE });

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
                target_tensor[grid_y][grid_x][offset_val + 0] = x_center * static_cast<float>(GRID_SIZE) - static_cast<float>(grid_x);
                target_tensor[grid_y][grid_x][offset_val + 1] = y_center * static_cast<float>(GRID_SIZE) - static_cast<float>(grid_y);
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