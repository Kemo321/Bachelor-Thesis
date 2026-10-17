#pragma once

#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/utils/SafeMath.hpp"

#include <cmath>
#include <map>
#include <string>

class Layer
{
public:
    float learning_rate = 0.001F;

    float gradient_clip = 0.0F;

    float momentum = 0.0F;

    float weight_decay = 0.0005F;

    bool use_adam = false;

    int adam_step = 0;

    virtual ~Layer() = default;

    virtual void train()
    {
        is_training_ = true;
    }

    virtual void eval()
    {
        is_training_ = false;
    }

    [[nodiscard]] virtual auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor = 0;

    [[nodiscard]] virtual auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor = 0;

    virtual void step(cudaStream_t stream = 0)
    {
        (void)stream;
    }

    virtual void clip_gradients(float abs_bound, cudaStream_t stream = 0)
    {
        (void)abs_bound;
        (void)stream;
    }

    [[nodiscard]] auto step_learning_rate() const -> float
    {
        return learning_rate;
    }

    [[nodiscard]] auto parameter_clip_bound() const -> float
    {
        return gradient_clip > 0.0F ? gradient_clip : 0.0F;
    }

    virtual auto get_parameters() -> std::map<std::string, dl::Tensor>
    {
        return {};
    }

    virtual void set_parameters(const std::map<std::string, dl::Tensor>& params)
    {
        (void)params;
    }

    virtual auto to(dl::Device device) -> void
    {
        device_ = device;
    }

protected:
    bool is_training_ = true;
    dl::Device device_ = dl::Device::GPU;
};
