#pragma once

#include "DeepLearnLib/core/Layer.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

class Gru : public Layer
{
public:
    Gru(int input_size, int hidden_size);
    ~Gru() override;

    Gru(const Gru&) = delete;
    auto operator=(const Gru&) -> Gru& = delete;

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;
    void step(cudaStream_t stream = 0) override;
    void clip_gradients(float abs_bound, cudaStream_t stream = 0) override;
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;
    auto to(dl::Device device) -> void override;

    struct LinearLayer
    {
        int rows = 0;
        int cols = 0;
        std::vector<float> matrix;
        std::vector<float> bias;
    };

    [[nodiscard]] auto export_linear_layers() const -> std::vector<LinearLayer>;
    [[nodiscard]] auto input_size() const -> int;
    [[nodiscard]] auto hidden_size() const -> int;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
