#pragma once

#include "DeepLearnLib/Precision.hpp"
#include "DeepLearnLib/SafeMath.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <cmath>
#include <map>
#include <string>

/**
 * @brief Abstract neural-network layer: forward/backward, optional SGD, device placement.
 *
 * Generic building block of DeepLearnLib. Concrete layers (Conv2d, FullyConnected,
 * FusedCBR2d, …) live in the core library. Application networks such as YOLO
 * compose these layers outside the core.
 *
 * Performance contract: implementations should reuse workspace tensors via
 * `dl::Tensor::ensure` and prefer in-place updates (`add_`, `sgd_update_`)
 * so a stable batch size does not allocate VRAM every step.
 */
class Layer
{
public:
    /** SGD step size applied in `step()`. */
    float learning_rate = 0.001F;
    /** Absolute per-parameter clip bound. `0` disables clipping. */
    float gradient_clip = 0.0F;
    /**
     * SGD momentum. `0` keeps vanilla SGD (`w -= lr * (g + wd w)`).
     * Non-zero uses a velocity buffer: `v = mu * v + (g + wd w); w -= lr * v`.
     */
    float momentum = 0.0F;
    /** L2 term added to the gradient before the SGD/momentum update. */
    float weight_decay = 0.0005F;

    virtual ~Layer() = default;

    /** Skip `step()` (and optionally skip backward through a frozen prefix). */
    void freeze()
    {
        frozen_ = true;
    }

    void unfreeze()
    {
        frozen_ = false;
    }

    [[nodiscard]] auto frozen() const -> bool
    {
        return frozen_;
    }

    /**
     * @brief Enable training behaviour (Dropout, BatchNorm running stats, …).
     */
    virtual void train()
    {
        is_training_ = true;
    }

    /**
     * @brief Enable evaluation behaviour (no dropout, frozen BN stats).
     */
    virtual void eval()
    {
        is_training_ = false;
    }

    /**
     * @brief Forward pass.
     * @param input_tensor Device tensor; layout is layer-specific (NCHW for conv).
     * @param stream CUDA stream. Empty/`0` uses the library current stream.
     * @return Output tensor. May alias an internal cache; do not free it.
     *
     * @note Implementations should write into a cached buffer when the shape is
     *       unchanged, to avoid `cudaMalloc` in the training loop.
     */
    [[nodiscard]] virtual auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor = 0;

    /**
     * @brief Backward pass.
     * @param output_error_derivative Gradient of the loss w.r.t. this layer's output.
     * @param stream CUDA stream.
     * @return Gradient w.r.t. the layer input (for the previous layer).
     *
     * @note Parameter gradients are accumulated on the layer; call `step()` after
     *       clipping if needed. Weight-decay is applied inside `sgd_update_`, not here.
     */
    [[nodiscard]] virtual auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor = 0;

    /**
     * @brief Apply the optimiser update to trainable parameters.
     * @param stream CUDA stream.
     *
     * @note Default is a no-op (activation layers). Parameterised layers typically
     *       call `Tensor::sgd_update_` in-place so no extra VRAM is allocated.
     */
    virtual void step(cudaStream_t stream = 0)
    {
        (void)stream;
    }

    /**
     * @brief Clip stored parameter gradients to `[-abs_bound, abs_bound]`.
     * @param abs_bound Absolute bound. Ignored when `<= 0`.
     * @param stream CUDA stream.
     *
     * @note Operates in-place on gradient buffers.
     */
    virtual void clip_gradients(float abs_bound, cudaStream_t stream = 0)
    {
        (void)abs_bound;
        (void)stream;
    }

    /**
     * @brief Learning rate divided by the current mixed-precision loss scale.
     * @return Effective SGD step size.
     */
    [[nodiscard]] auto scaled_learning_rate() const -> float
    {
        const float scale = dl::loss_scale();
        return learning_rate / fmaxf(scale, dl::kSafeEps);
    }

    /**
     * @brief Gradient clip bound scaled for mixed precision, or `0` if disabled.
     */
    [[nodiscard]] auto parameter_clip_bound() const -> float
    {
        return gradient_clip > 0.0F ? dl::scaled_gradient_clip(gradient_clip) : 0.0F;
    }

    /**
     * @brief Named trainable tensors for serialisation.
     * @return Map of parameter name to tensor (empty for layers without weights).
     */
    virtual auto get_parameters() -> std::map<std::string, dl::Tensor>
    {
        return {};
    }

    /**
     * @brief Restore trainable tensors from `get_parameters()`-compatible names.
     * @param params Name to tensor map. Unknown names are ignored by implementations.
     */
    virtual void set_parameters(const std::map<std::string, dl::Tensor>& params)
    {
        (void)params;
    }

    /**
     * @brief Move layer state to @p device (GPU is the training path).
     * @param device Target device.
     */
    virtual auto to(dl::Device device) -> void
    {
        device_ = device;
    }

protected:
    bool is_training_ = true;
    bool frozen_ { false };
    dl::Device device_ = dl::Device::GPU;
};
