#pragma once

#include "experiment_config.hpp"

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <memory>
#include <vector>

/**
 * Puts every layer on the GPU and installs the optimiser fields both stacks share.
 * Adam ignores momentum, so the custom path stores 0 and does not build an SGD velocity.
 */
inline auto configure_custom_layers(const std::vector<std::shared_ptr<Layer>>& layers, float learning_rate,
    float momentum, float weight_decay, float gradient_clip, bool adam) -> void
{
    for (const auto& layer : layers)
    {
        layer->to(dl::Device::GPU);
        layer->use_adam = adam;
        layer->gradient_clip = gradient_clip;
        layer->train();
    }
    apply_sgd_hyperparameters(layers, learning_rate, adam ? 0.0F : momentum, weight_decay);
}

inline auto backward_and_step(const std::vector<std::shared_ptr<Layer>>& layers, dl::Tensor grad, cudaStream_t stream)
    -> void
{
    for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
    {
        grad = (*iterator)->backward(grad, stream);
    }
    for (const auto& layer : layers)
    {
        layer->step(stream);
    }
}
