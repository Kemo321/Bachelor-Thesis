#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <torch/torch.h>

#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/dataset.hpp"

auto get_current_time_sec() -> double
{
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

auto get_yolo_lr = [](int epoch) -> float {
    return 1e-6F;
};

auto main(int argc, char* argv[]) -> int
{
    const int batch_size = 32;
    const int log_interval_images = 3000;
    const int total_epochs = 60;
    const std::string data_root = "../data/VOCdevkit";
    const std::string csv_log_path = "training_metrics_custom.csv";

    std::cout << "========================================\n";
    std::cout << "[INIT] Starting YOLO training pipeline (Custom)\n";
    std::cout << "========================================\n";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "[INFO] Device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root + "/VOC2012", train_paths, val_paths, test_paths);

    if (train_paths.images.empty() || test_paths.images.empty())
    {
        std::cerr << "[ERROR] Missing training or test data!\n";
        return -1;
    }

    auto train_loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths, true).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));

    auto test_loader = torch::data::make_data_loader(
        VOCYoloDataset(test_paths, false).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

    YOLO custom_model;
    
    for (const auto& layer_pointer : custom_model->get_all_layers())
    {
        layer_pointer->to(device);
    }

    Network trainer(custom_model->get_all_layers(), get_yolo_lr(1));
    std::cout << "[INFO] Custom model created.\n";

    std::ofstream csv_file(csv_log_path);
    csv_file << "Epoch,TotalTime_Sec,ImagesPerSec,TrainLoss,TestLoss\n";

    double global_start_time = get_current_time_sec();
    float best_test_loss = std::numeric_limits<float>::max();

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        float current_lr = get_yolo_lr(epoch);
        for (const auto& layer_pointer : custom_model->get_all_layers()) {
            layer_pointer->learning_rate = current_lr;
        }

        std::cout << "\n--- EPOCH " << epoch << " / " << total_epochs << " [LR: " << current_lr << "] ---\n";
        float epoch_train_loss = 0.0F;
        int processed_epoch = 0;
        int images_since_last_log = 0;

        double log_start_time = get_current_time_sec();

        {
            for (auto& batch : *train_loader)
            {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);

                auto pred = custom_model->forward(data);

                float batch_loss = YOLOLoss::loss(target, pred).item<float>();
                auto grad_error = YOLOLoss::loss_derivative(target, pred);

                grad_error = grad_error.clamp(-1.0, 1.0);

                auto layers = custom_model->get_all_layers();
                for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
                {
                    grad_error = (*iterator)->backward(grad_error);
                }

                if (device.is_cuda())
                {
                    torch::cuda::synchronize();
                }

                epoch_train_loss += batch_loss;

                int current_batch_size = data.size(0);
                processed_epoch += current_batch_size;
                images_since_last_log += current_batch_size;

                if (images_since_last_log >= log_interval_images)
                {
                    double now = get_current_time_sec();
                    double elapsed = now - log_start_time;
                    double fps = images_since_last_log / elapsed;

                    std::cout << "[TRAIN] Processed: " << std::setw(5) << processed_epoch
                              << " / " << train_paths.images.size()
                              << " | Loss: " << std::fixed << std::setprecision(4) << batch_loss
                              << " | Throughput: " << std::setprecision(1) << fps << " img/s\n";

                    images_since_last_log = 0;
                    log_start_time = get_current_time_sec();
                }
            }
        }

        float avg_train_loss = epoch_train_loss / static_cast<float>(train_paths.images.size() / batch_size + 1);

        std::cout << "[TEST] Evaluating model on the test set...\n";
        float test_loss_sum = 0.0F;
        int test_batches = 0;

        {
            torch::NoGradGuard no_grad;
            for (auto& batch : *test_loader)
            {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);
                auto pred = custom_model->forward(data);

                test_loss_sum += YOLOLoss::loss(target, pred).item<float>();
                test_batches++;
            }
        }

        float avg_test_loss = test_batches > 0 ? (test_loss_sum / static_cast<float>(test_batches)) : 0.0F;

        double current_time = get_current_time_sec();
        double epoch_end_time = current_time - global_start_time;
        double overall_fps = processed_epoch / (current_time - (log_start_time)); 

        std::cout << "[EPOCH SUMMARY " << epoch << "]\n"
              << " -> Average Training Loss: " << avg_train_loss << "\n"
              << " -> Average Test Loss:     " << avg_test_loss << "\n";

        csv_file << epoch << ","
                 << epoch_end_time << ","
                 << overall_fps << ","
                 << avg_train_loss << ","
                 << avg_test_loss << "\n";
        csv_file.flush();

        if (avg_test_loss < best_test_loss)
        {
            best_test_loss = avg_test_loss;
            std::string best_model_name = "custom_yolo_best_epoch.pt";
            trainer.save(best_model_name);
            std::cout << "[SAVE] Saved new best model: " << best_model_name << "\n";
        }

        trainer.save("custom_yolo_latest.pt");
    }

    std::cout << "========================================\n";
    std::cout << "[DONE] Training completed successfully.\n";
    std::cout << "Best result on test set: " << best_test_loss << "\n";
    std::cout << "Logs saved to file: " << csv_log_path << "\n";
    std::cout << "========================================\n";

    return 0;
}