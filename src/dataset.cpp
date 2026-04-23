#include "DeepLearnLib/dataset.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <pugixml.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace {
    constexpr int GRID_SIZE = 7;
    constexpr int NUM_CLASSES = 20;
    constexpr int BBOX_OUTPUT_SIZE = 30;
    constexpr int BOXES_PER_CELL = 2;
    constexpr int BOX_PARAMS = 5;
    constexpr int CLASS_OFFSET = 10;
    constexpr float NORMALIZATION_FACTOR = 255.0F;
    constexpr float CENTER_DIVISOR = 2.0F;

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

    auto convert_voc_to_yolo(const std::string& annot_dir, const std::string& label_dir, 
                             const std::string& jpeg_dir) -> void
    {
        int converted = 0;
        for (const auto& entry : std::filesystem::directory_iterator(annot_dir))
        {
            if (entry.path().extension() != ".xml") {
                continue;
            }

            std::string base = entry.path().stem().string();
            std::string xml_path = entry.path().string();
            std::string txt_path = label_dir + "/" + base + ".txt";
            std::string img_path = jpeg_dir + "/" + base + ".jpg";

            if (!std::filesystem::exists(img_path)) {
                continue;
            }

            pugi::xml_document doc;
            if (!doc.load_file(xml_path.c_str())) {
                continue;
            }

            auto root = doc.child("annotation");
            auto size = root.child("size");
            int width = size.child("width").text().as_int();
            int height = size.child("height").text().as_int();
            if (width == 0 || height == 0) {
                continue;
            }

            std::ofstream txt(txt_path);
            for (auto obj = root.child("object"); obj != nullptr; obj = obj.next_sibling("object"))
            {
                std::string name = obj.child("name").text().as_string();
                auto bndbox = obj.child("bndbox");
                if (bndbox.empty()) {
                    continue;
                }

                int cls = voc_class_to_id(name);
                if (cls >= 0)
                {
                    float xmin = bndbox.child("xmin").text().as_float();
                    float ymin = bndbox.child("ymin").text().as_float();
                    float xmax = bndbox.child("xmax").text().as_float();
                    float ymax = bndbox.child("ymax").text().as_float();

                    float x_center = ((xmin + xmax) / CENTER_DIVISOR) / static_cast<float>(width);
                    float y_center = ((ymin + ymax) / CENTER_DIVISOR) / static_cast<float>(height);
                    float box_width = (xmax - xmin) / static_cast<float>(width);
                    float box_height = (ymax - ymin) / static_cast<float>(height);

                    txt << cls << " " << x_center << " " << y_center << " " << box_width << " " << box_height << "\n";
                }
            }
            converted++;
        }
        if (converted > 0)
        {
            std::cout << "[INFO] Konwersja XML -> YOLO: " << converted << " plikow .txt utworzonych\n";
        }
    }
}

void split_dataset(const std::string& voc_root,
    DataPaths& train, DataPaths& val, DataPaths& test,
    float train_ratio, float val_ratio)
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

    bool need_convert = false;
    for (const auto& entry : std::filesystem::directory_iterator(jpeg_dir))
    {
        if (entry.path().extension() == ".jpg" && !std::filesystem::exists(label_dir + "/" + entry.path().stem().string() + ".txt"))
        {
            need_convert = true;
            break;
        }
    }
    if (need_convert)
    {
        convert_voc_to_yolo(annot_dir, label_dir, jpeg_dir);
    }

    std::vector<std::pair<std::string, std::string>> all_pairs;
    for (const auto& entry : std::filesystem::directory_iterator(jpeg_dir))
    {
        if (entry.path().extension() == ".jpg")
        {
            std::string img = entry.path().string();
            std::string base = entry.path().stem().string();
            std::string lbl = label_dir + "/" + base + ".txt";
            if (std::filesystem::exists(lbl))
            {
                all_pairs.emplace_back(img, lbl);
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

    size_t total = all_pairs.size();
    auto train_end = static_cast<size_t>(static_cast<float>(total) * train_ratio);
    auto val_end = train_end + static_cast<size_t>(static_cast<float>(total) * val_ratio);

    for (size_t i = 0; i < total; ++i)
    {
        if (i < train_end)
        {
            train.images.push_back(all_pairs[i].first);
            train.labels.push_back(all_pairs[i].second);
        }
        else if (i < val_end)
        {
            val.images.push_back(all_pairs[i].first);
            val.labels.push_back(all_pairs[i].second);
        }
        else
        {
            test.images.push_back(all_pairs[i].first);
            test.labels.push_back(all_pairs[i].second);
        }
    }

    std::cout << "[INFO] Podzial zakonczony! Razem: " << total << " obrazow\n";
    std::cout << "   Train: " << train.images.size() << " | Val: " << val.images.size()
              << " | Test: " << test.images.size() << "\n\n";
}

VOCYoloDataset::VOCYoloDataset(const DataPaths& p) : paths(p) {}

auto VOCYoloDataset::get(size_t index) -> torch::data::Example<>
{
    int width{};
    int height{};
    int channels{};
    unsigned char* data = stbi_load(paths.images[index].c_str(), &width, &height, &channels, 3);
    if (data == nullptr)
    {
        return { torch::zeros({ 3, img_size, img_size }), torch::zeros({ GRID_SIZE, GRID_SIZE, BBOX_OUTPUT_SIZE }) };
    }

    auto img = torch::from_blob(data, { height, width, 3 }, torch::kUInt8).clone();
    stbi_image_free(data);

    img = img.to(torch::kFloat32).div_(NORMALIZATION_FACTOR).permute({ 2, 0, 1 });
    img = torch::nn::functional::interpolate(
              img.unsqueeze(0),
              torch::nn::functional::InterpolateFuncOptions()
                  .size(std::vector<int64_t> { img_size, img_size })
                  .mode(torch::kBilinear)
                  .align_corners(false))
              .squeeze(0);

    torch::Tensor target = torch::zeros({ GRID_SIZE, GRID_SIZE, BBOX_OUTPUT_SIZE });

    std::ifstream file(paths.labels[index]);
    int cls{};
    float x_center{};
    float y_center{};
    float box_width{};
    float box_height{};
    while (file >> cls >> x_center >> y_center >> box_width >> box_height)
    {
        int grid_x = static_cast<int>(x_center * GRID_SIZE);
        int grid_y = static_cast<int>(y_center * GRID_SIZE);
        if (grid_x < 0 || grid_x >= GRID_SIZE || grid_y < 0 || grid_y >= GRID_SIZE) {
            continue;
        }

        for (int box_idx = 0; box_idx < BOXES_PER_CELL; ++box_idx)
        {
            int offset = box_idx * BOX_PARAMS;
            if (target[grid_y][grid_x][offset + 4].item<float>() == 0.0F)
            {
                target[grid_y][grid_x][offset + 0] = x_center * GRID_SIZE - static_cast<float>(grid_x);
                target[grid_y][grid_x][offset + 1] = y_center * GRID_SIZE - static_cast<float>(grid_y);
                target[grid_y][grid_x][offset + 2] = box_width;
                target[grid_y][grid_x][offset + 3] = box_height;
                target[grid_y][grid_x][offset + 4] = 1.0F;
                target[grid_y][grid_x][CLASS_OFFSET + cls] = 1.0F;
                break;
            }
        }
    }
    return { img, target };
}

auto VOCYoloDataset::size() const -> torch::optional<size_t> { return paths.images.size(); }