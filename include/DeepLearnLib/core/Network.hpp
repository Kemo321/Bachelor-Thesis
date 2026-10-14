#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/utils/SafeMath.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

/**
 * @brief Ordered stack of generic layers with binary weight I/O.
 *
 * `Network` does not know about a particular model. Applications pass their
 * layers in via `get_all_layers()` and own the training loop, so the loss and
 * the learning-rate schedule stay outside this class.
 */
class Network
{
public:
    /**
     * @brief Take ownership of an ordered layer list and broadcast optimiser state.
     * @param layers_vector Non-null layers in forward order.
     * @param learning_rate SGD step size copied onto every layer.
     * @param gradient_clip Absolute parameter-gradient clip (`0` disables).
     */
    Network(std::vector<std::shared_ptr<Layer>> layers_vector, float learning_rate,
        float gradient_clip = dl::kDefaultGradientClip);

    /**
     * @brief Sequential forward through all layers.
     * @param input_tensor Device input.
     * @param stream CUDA stream.
     * @return Final layer output. May alias a layer cache.
     */
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;

    /**
     * @brief Write layer parameters to a custom binary checkpoint.
     * @param path Destination file.
     */
    void save(const std::string& path);

    /**
     * @brief Load parameters written by `save()`.
     * @param path Source file.
     */
    void load(const std::string& path);

    /**
     * @brief Update the clip bound and push it to every layer.
     * @param abs_bound Absolute bound; `<= 0` disables clipping.
     */
    void set_gradient_clip(float abs_bound);

    /** @return Current absolute parameter-gradient clip bound. */
    [[nodiscard]] auto gradient_clip() const -> float;

    /**
     * @brief Clip a loss gradient tensor, reusing an internal buffer.
     * @param gradient Incoming dL/dpred.
     * @return Clipped tensor (cached; do not assume unique ownership).
     *
     * @note No allocation when the gradient shape is stable.
     */
    [[nodiscard]] auto clip_loss_gradient(const dl::Tensor& gradient) const -> dl::Tensor;

    /**
     * @brief Clip stored parameter gradients on every layer.
     * @param stream CUDA stream.
     */
    void clip_parameter_gradients(cudaStream_t stream = 0);

private:
    auto sync_layer_optimizer_state() -> void;

    std::vector<std::shared_ptr<Layer>> layers_;
    float gradient_clip_ { dl::kDefaultGradientClip };
    mutable std::optional<dl::Tensor> loss_grad_clip_cache_;
};
