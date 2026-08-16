#pragma once

#include "DeepLearnLib/Tensor.hpp"
#include <map>
#include <string>

/**
 * Abstract neural-network layer: forward/backward, optional SGD step, and device placement.
 */
class Layer
{
public:
    float learning_rate = 0.001F;

    virtual ~Layer() = default;

    virtual void train()
    {
        is_training_ = true;
    }

    virtual void eval()
    {
        is_training_ = false;
    }

    [[nodiscard]] virtual auto forward(const dl::Tensor& input_tensor) -> dl::Tensor = 0;
    [[nodiscard]] virtual auto backward(const dl::Tensor& output_error_derivative) -> dl::Tensor = 0;

    virtual void step() { }

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
