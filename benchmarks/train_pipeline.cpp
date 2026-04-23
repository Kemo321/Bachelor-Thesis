#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <limits>
#include <string>
#include <torch/torch.h>

#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/model.hpp"

// Pomocnicza funkcja do pobierania aktualnego czasu w sekundach (double)
double get_current_time_sec() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

int main(int argc, char* argv[])
{
    // --- KONFIGURACJA POTOKU ---
    const int batch_size = 16; // Zwiększ do 16/32 dla RTX 5080, jeśli pamięć pozwoli
    const int log_interval_images = 1000;
    const double max_training_time_seconds = 60.0 * 60.0; // 1 godzina
    const std::string data_root = "../data/VOCdevkit";
    const std::string csv_log_path = "training_metrics.csv";

    std::cout << "========================================\n";
    std::cout << "[INIT] Rozpoczynanie potoku treningowego YOLOv1\n";
    std::cout << "========================================\n";

    // 1. Urządzenie
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "[INFO] Urzadzenie: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    // 2. Dane
    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root + "/VOC2012", train_paths, val_paths, test_paths);

    if (train_paths.images.empty() || test_paths.images.empty())
    {
        std::cerr << "[BLAD] Brak danych treningowych lub testowych!\n";
        return -1;
    }

    auto train_loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));

    auto test_loader = torch::data::make_data_loader(
        VOCYoloDataset(test_paths).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

    // 3. Model i Optymalizator
    YOLOv1 model;
    model->to(device);
    std::cout << "[INFO] Model utworzony i załadowany do pamieci VRAM.\n";
    
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-4));

    // 4. Przygotowanie pliku CSV do dokumentacji
    std::ofstream csv_file(csv_log_path);
    csv_file << "Epoch,TotalTime_Sec,ImagesPerSec,TrainLoss,TestLoss\n";

    // Zmienne śledzące czas i postęp
    double global_start_time = get_current_time_sec();
    float best_test_loss = std::numeric_limits<float>::max();
    int epoch = 1;

    // --- GŁÓWNA PĘTLA TRENINGOWA ---
    while (true)
    {
        double current_time = get_current_time_sec() - global_start_time;
        if (current_time >= max_training_time_seconds)
        {
            std::cout << "\n[INFO] Osiagnieto limit czasu (" << max_training_time_seconds / 60 << " min). Koniec trenowania.\n";
            break;
        }

        std::cout << "\n--- EPOKA " << epoch << " ---\n";
        model->train();
        float epoch_train_loss = 0.0f;
        int processed_epoch = 0;
        int images_since_last_log = 0;
        
        double log_start_time = get_current_time_sec();

        // --- PĘTLA BATCHY (TRENING) ---
        for (auto& batch : *train_loader)
        {
            auto data = batch.data.to(device, true);
            auto target = batch.target.to(device, true);

            optimizer.zero_grad();
            auto pred = model->forward(data);
            auto loss = compute_yolo_loss(pred, target);

            loss.backward();
            optimizer.step();

            if (device.is_cuda()) torch::cuda::synchronize();

            float batch_loss = loss.item<float>();
            epoch_train_loss += batch_loss;
            
            int current_batch_size = data.size(0);
            processed_epoch += current_batch_size;
            images_since_last_log += current_batch_size;

            // Logowanie co 1000 obrazów
            if (images_since_last_log >= log_interval_images)
            {
                double now = get_current_time_sec();
                double elapsed = now - log_start_time;
                double fps = images_since_last_log / elapsed;

                std::cout << "[TRENING] Przetworzono: " << std::setw(5) << processed_epoch 
                          << " / " << train_paths.images.size() 
                          << " | Loss: " << std::fixed << std::setprecision(4) << batch_loss 
                          << " | Przepustowosc: " << std::setprecision(1) << fps << " img/s\n";

                // Reset liczników logowania
                images_since_last_log = 0;
                log_start_time = get_current_time_sec();
            }
        }
        
        // Średni błąd treningowy w epoce
        float avg_train_loss = epoch_train_loss / (train_paths.images.size() / batch_size + 1);

        // --- EWALUACJA (TEST) ---
        std::cout << "[TEST] Trwa testowanie modelu na zbiorze testowym...\n";
        model->eval();
        float test_loss_sum = 0.0f;
        int test_batches = 0;
        
        torch::NoGradGuard no_grad; // Wyłącza śledzenie gradientów (oszczędza pamięć i czas)
        
        for (auto& batch : *test_loader)
        {
            auto data = batch.data.to(device, true);
            auto target = batch.target.to(device, true);
            auto pred = model->forward(data);
            
            test_loss_sum += compute_yolo_loss(pred, target).item<float>();
            test_batches++;
        }
        
        float avg_test_loss = test_batches > 0 ? (test_loss_sum / test_batches) : 0.0f;

        // --- PODSUMOWANIE EPOKI I ZAPIS ---
        double epoch_end_time = get_current_time_sec() - global_start_time;
        double overall_fps = processed_epoch / (epoch_end_time - current_time);

        std::cout << "[PODSUMOWANIE EPOKI " << epoch << "]\n"
                  << " -> Sredni Loss Treningowy: " << avg_train_loss << "\n"
                  << " -> Sredni Loss Testowy:    " << avg_test_loss << "\n"
                  << " -> Sredni czas epoki FPS:  " << overall_fps << " img/s\n";

        // Zapis do CSV
        csv_file << epoch << "," 
                 << epoch_end_time << "," 
                 << overall_fps << "," 
                 << avg_train_loss << "," 
                 << avg_test_loss << "\n";
        csv_file.flush(); // Wymuszenie zapisu na dysk

        // Zapis modelu (jeśli jest najlepszy dotychczas)
        if (avg_test_loss < best_test_loss)
        {
            best_test_loss = avg_test_loss;
            std::string best_model_name = "yolov1_best_epoch_" + std::to_string(epoch) + ".pt";
            torch::save(model, best_model_name);
            std::cout << "[ZAPIS] Zapisano nowy najlepszy model: " << best_model_name << "\n";
        }
        
        // Regularny zapis na koniec epoki
        torch::save(model, "yolov1_latest.pt");

        epoch++;
    }

    std::cout << "========================================\n";
    std::cout << "[KONIEC] Trening zakonczony sukcesem.\n";
    std::cout << "Najlepszy wynik na zbiorze testowym: " << best_test_loss << "\n";
    std::cout << "Logi zapisano do pliku: " << csv_log_path << "\n";
    std::cout << "========================================\n";

    return 0;
}