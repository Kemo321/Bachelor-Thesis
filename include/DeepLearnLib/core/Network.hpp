#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/utils/SafeMath.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

class Network
{
public:
    Network(std::vector<std::shared_ptr<Layer>> layers_vector, float learning_rate,
        float gradient_clip = dl::kDefaultGradientClip);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;

    void save(const std::string& path);

    void load(const std::string& path);

    void set_gradient_clip(float abs_bound);

    [[nodiscard]] auto gradient_clip() const -> float;

    [[nodiscard]] auto clip_loss_gradient(const dl::Tensor& gradient) const -> dl::Tensor;

    void clip_parameter_gradients(cudaStream_t stream = 0);

private:
    auto sync_layer_optimizer_state() -> void;

    std::vector<std::shared_ptr<Layer>> layers_;
    float gradient_clip_ { dl::kDefaultGradientClip };
    mutable std::optional<dl::Tensor> loss_grad_clip_cache_;
};
