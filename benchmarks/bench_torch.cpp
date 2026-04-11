#include <algorithm>
#include <benchmark/benchmark.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <pugixml.hpp>

struct DataPaths
{
    std::vector<std::string> images;
    std::vector<std::string> labels;
};

int voc_class_to_id(const std::string& class_name)
{
    static const std::unordered_map<std::string, int> voc_classes = {
        { "aeroplane", 0 }, { "bicycle", 1 }, { "bird", 2 }, { "boat", 3 }, { "bottle", 4 },
        { "bus", 5 }, { "car", 6 }, { "cat", 7 }, { "chair", 8 }, { "cow", 9 },
        { "diningtable", 10 }, { "dog", 11 }, { "horse", 12 }, { "motorbike", 13 },
        { "person", 14 }, { "pottedplant", 15 }, { "sheep", 16 }, { "sofa", 17 },
        { "train", 18 }, { "tvmonitor", 19 }
    };
    auto it = voc_classes.find(class_name);
    return (it != voc_classes.end()) ? it->second : -1;
}

void convert_voc_to_yolo(const std::string& annot_dir, const std::string& label_dir, const std::string& jpeg_dir)
{
    int converted = 0;
    for (const auto& entry : std::filesystem::directory_iterator(annot_dir))
    {
        if (entry.path().extension() != ".xml")
        {
            continue;
        }

        std::string base = entry.path().stem().string();
        std::string xml_path = entry.path().string();
        std::string txt_path = label_dir + "/" + base + ".txt";
        std::string img_path = jpeg_dir + "/" + base + ".jpg";

        if (!std::filesystem::exists(img_path))
        {
            continue;
        }

        pugi::xml_document doc;
        if (!doc.load_file(xml_path.c_str()))
        {
            continue;
        }

        auto root = doc.child("annotation");
        auto size = root.child("size");
        int width = size.child("width").text().as_int();
        int height = size.child("height").text().as_int();
        if (width == 0 || height == 0)
        {
            continue;
        }

        std::ofstream txt(txt_path);
        for (auto obj = root.child("object"); obj != nullptr; obj = obj.next_sibling("object"))
        {
            std::string name = obj.child("name").text().as_string();
            auto bndbox = obj.child("bndbox");
            if (bndbox.empty())
            {
                continue;
            }

            int cls = voc_class_to_id(name);
            if (cls >= 0)
            {
                float xmin = bndbox.child("xmin").text().as_float();
                float ymin = bndbox.child("ymin").text().as_float();
                float xmax = bndbox.child("xmax").text().as_float();
                float ymax = bndbox.child("ymax").text().as_float();

                float x_center = ((xmin + xmax) / 2.0F) / width;
                float y_center = ((ymin + ymax) / 2.0F) / height;
                float w = (xmax - xmin) / width;
                float h = (ymax - ymin) / height;

                txt << cls << " " << x_center << " " << y_center << " " << w << " " << h << "\n";
            }
        }
        converted++;
    }
    if (converted > 0)
    {
        std::cout << "[INFO] Konwersja XML → YOLO: " << converted << " plików .txt utworzonych\n";
    }
}

void split_dataset(const std::string& voc_root,
    DataPaths& train, DataPaths& val, DataPaths& test,
    float train_ratio = 0.7F, float val_ratio = 0.15F)
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
    for (const auto& e : std::filesystem::directory_iterator(jpeg_dir))
    {
        if (e.path().extension() == ".jpg" && !std::filesystem::exists(label_dir + "/" + e.path().stem().string() + ".txt"))
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
        std::cerr << "[BLAD] Brak sparowanych obrazów i etykiet!\n";
        return;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all_pairs.begin(), all_pairs.end(), g);

    size_t total = all_pairs.size();
    size_t train_end = static_cast<size_t>(total * train_ratio);
    size_t val_end = train_end + static_cast<size_t>(total * val_ratio);

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

class VOCYoloDataset : public torch::data::datasets::Dataset<VOCYoloDataset>
{
private:
    DataPaths paths;
    const int img_size = 448;

public:
    explicit VOCYoloDataset(const DataPaths& p)
        : paths(p)
    {
    }

    torch::data::Example<> get(size_t index) override
    {
        int w;
        int h;
        int c;
        unsigned char* data = stbi_load(paths.images[index].c_str(), &w, &h, &c, 3);
        if (data == nullptr)
        {
            return { torch::zeros({ 3, img_size, img_size }), torch::zeros({ 7, 7, 30 }) };
        }

        auto img = torch::from_blob(data, { h, w, 3 }, torch::kUInt8).clone();
        stbi_image_free(data);

        img = img.to(torch::kFloat32).div_(255.0F).permute({ 2, 0, 1 });

        img = torch::nn::functional::interpolate(
            img.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t> { img_size, img_size })
                .mode(torch::kBilinear)
                .align_corners(false))
                  .squeeze(0);

        torch::Tensor target = torch::zeros({ 7, 7, 30 });

        std::ifstream file(paths.labels[index]);
        int cls;
        float xc;
        float yc;
        float ww;
        float hh;
        while (file >> cls >> xc >> yc >> ww >> hh)
        {
            int gx = static_cast<int>(xc * 7);
            int gy = static_cast<int>(yc * 7);
            if (gx < 0 || gx >= 7 || gy < 0 || gy >= 7)
            {
                continue;
            }

            for (int b = 0; b < 2; ++b)
            {
                int off = b * 5;
                if (target[gy][gx][off + 4].item<float>() == 0.0F)
                {
                    target[gy][gx][off + 0] = xc * 7.0F - gx;
                    target[gy][gx][off + 1] = yc * 7.0F - gy;
                    target[gy][gx][off + 2] = ww;
                    target[gy][gx][off + 3] = hh;
                    target[gy][gx][off + 4] = 1.0F;
                    target[gy][gx][10 + cls] = 1.0F;
                    break;
                }
            }
        }
        return { img, target };
    }

    torch::optional<size_t> size() const override { return paths.images.size(); }
};

struct YOLOv1Impl : torch::nn::Module
{
    torch::nn::Sequential backbone { nullptr };
    torch::nn::Sequential head { nullptr };

    YOLOv1Impl()
    {
        using namespace torch::nn;

        backbone = register_module("backbone", Sequential(Conv2d(Conv2dOptions(3, 64, 7).stride(2).padding(3)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), MaxPool2d(MaxPool2dOptions(2).stride(2)),

                                                   Conv2d(Conv2dOptions(64, 192, 3).padding(1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), MaxPool2d(MaxPool2dOptions(2).stride(2)),

                                                   Conv2d(Conv2dOptions(192, 128, 1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(128, 256, 3).padding(1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(256, 256, 1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(256, 512, 3).padding(1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), MaxPool2d(MaxPool2dOptions(2).stride(2)),

                                                   Conv2d(Conv2dOptions(512, 256, 1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(256, 512, 3).padding(1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(512, 256, 1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(256, 512, 3).padding(1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(512, 512, 1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(512, 1024, 3).padding(1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), MaxPool2d(MaxPool2dOptions(2).stride(2)),

                                                   Conv2d(Conv2dOptions(1024, 512, 1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(512, 1024, 3).padding(1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(1024, 512, 1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(512, 1024, 3).padding(1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(1024, 1024, 3).padding(1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(1024, 1024, 3).stride(2).padding(1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(1024, 1024, 3).padding(1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Conv2d(Conv2dOptions(1024, 1024, 3).padding(1)), LeakyReLU(LeakyReLUOptions().negative_slope(0.1))));

        head = register_module("head", Sequential(Linear(7 * 7 * 1024, 4096), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Dropout(DropoutOptions(0.5)), Linear(4096, 7 * 7 * 30)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = backbone->forward(x);
        x = x.view({ x.size(0), -1 });
        x = head->forward(x);
        return x.view({ -1, 7, 7, 30 });
    }
};
TORCH_MODULE(YOLOv1);

torch::Tensor compute_yolo_loss(const torch::Tensor& pred, const torch::Tensor& target)
{
    const float lambda_coord = 5.0F;
    const float lambda_noobj = 0.5F;

    auto obj_mask = target.slice(3, 4, 5) > 0.5F;
    auto noobj_mask = target.slice(3, 4, 5) <= 0.5F;

    auto pred_boxes = pred.slice(3, 0, 10);
    auto target_boxes = target.slice(3, 0, 10);

    auto loss_xy = torch::mse_loss(
        pred_boxes.masked_select(obj_mask.expand({ -1, -1, -1, 10 })),
        target_boxes.masked_select(obj_mask.expand({ -1, -1, -1, 10 })));

    auto pred_wh = torch::sqrt(torch::clamp(pred_boxes.slice(3, 2, 4), 1e-6F));
    auto target_wh = torch::sqrt(torch::clamp(target_boxes.slice(3, 2, 4), 1e-6F));
    auto loss_wh = torch::mse_loss(
        pred_wh.masked_select(obj_mask.expand({ -1, -1, -1, 2 })),
        target_wh.masked_select(obj_mask.expand({ -1, -1, -1, 2 })));

    auto pred_conf = pred.slice(3, 4, 5);
    auto target_conf = target.slice(3, 4, 5);
    auto loss_conf_obj = torch::mse_loss(pred_conf.masked_select(obj_mask), target_conf.masked_select(obj_mask));
    auto loss_conf_noobj = torch::mse_loss(pred_conf.masked_select(noobj_mask), target_conf.masked_select(noobj_mask));

    auto pred_class = pred.slice(3, 10, 30);
    auto target_class = target.slice(3, 10, 30);
    auto loss_class = torch::mse_loss(
        pred_class.masked_select(obj_mask.expand({ -1, -1, -1, 20 })),
        target_class.masked_select(obj_mask.expand({ -1, -1, -1, 20 })));

    return lambda_coord * (loss_xy + loss_wh) + loss_conf_obj + lambda_noobj * loss_conf_noobj + loss_class;
}

static void BM_YOLOv1_Full_Training(benchmark::State& state)
{
    const int batch_size = state.range(0);
    const std::string data_root = "../data/VOCdevkit";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "[INFO] Urzadzenie: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    DataPaths train_paths;
    DataPaths val_paths;
    DataPaths test_paths;
    split_dataset(data_root + "/VOC2012", train_paths, val_paths, test_paths);

    if (train_paths.images.empty())
    {
        state.SkipWithError("Brak danych!");
        return;
    }

    auto train_loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));

    auto val_loader = torch::data::make_data_loader(
        VOCYoloDataset(val_paths).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

    YOLOv1 model;
    model->to(device);
    std::cout << "[INFO] Model YOLOv1 utworzony i przeniesiony na urzadzenie\n";
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-4));

    float best_val_loss = std::numeric_limits<float>::max();

    for (auto _ : state)
    {
        std::cout << "[INFO] Iteracja " << state.iterations() + 1 << " rozpoczęta\n";
        float epoch_loss = 0.0F;
        model->train();

        for (auto& batch : *train_loader)
        {
            std::cout << "[INFO] Przetwarzanie batcha " << batch.data.size(0) << " obrazow\n";
            auto data = batch.data.to(device, true);
            auto target = batch.target.to(device, true);

            optimizer.zero_grad();
            auto pred = model->forward(data);
            auto loss = compute_yolo_loss(pred, target);

            loss.backward();
            optimizer.step();
            if (device.is_cuda())
            {
                torch::cuda::synchronize();
            }

            epoch_loss += loss.item<float>();
        }

        if (state.iterations() % 5 == 0)
        {
            model->eval();
            float val_sum = 0.0F;
            int cnt = 0;
            torch::NoGradGuard no_grad;
            for (auto& batch : *val_loader)
            {
                auto data = batch.data.to(device, true);
                auto target = batch.target.to(device, true);
                auto pred = model->forward(data);
                val_sum += compute_yolo_loss(pred, target).item<float>();
                cnt++;
            }
            if (cnt > 0)
            {
                float val_loss = val_sum / cnt;
                if (val_loss < best_val_loss)
                {
                    best_val_loss = val_loss;
                }
            }
        }
        benchmark::DoNotOptimize(epoch_loss);
    }

    state.counters["Best_Val_Loss"] = best_val_loss;
    state.counters["Batch_Size"] = batch_size;
    state.counters["Train_Images"] = static_cast<int64_t>(train_paths.images.size());
}

BENCHMARK(BM_YOLOv1_Full_Training)
    ->Arg(8)
    ->Iterations(1)
    ->Unit(benchmark::kSecond)
    ->UseRealTime();

BENCHMARK_MAIN();
