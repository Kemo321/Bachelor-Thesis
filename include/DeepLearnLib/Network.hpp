#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/SafeMath.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

#include <memory>
#include <string>
#include <vector>

/**
 * Ordered stack of custom layers with a training loop and binary weight I/O.
 */
class Network
{
public:
    Network(std::vector<std::shared_ptr<Layer>> layers_vector, float learning_rate,
        float gradient_clip = dl::kDefaultGradientClip);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;
    void fit(const dl::Tensor& x_train, const dl::Tensor& y_train, int epochs, int verbose = 1);
    void save(const std::string& path);
    void load(const std::string& path);

    void set_gradient_clip(float abs_bound);
    [[nodiscard]] auto gradient_clip() const -> float;
    [[nodiscard]] auto clip_loss_gradient(const dl::Tensor& gradient) const -> dl::Tensor;
    void clip_parameter_gradients(cudaStream_t stream = 0);

private:
    std::vector<std::shared_ptr<Layer>> layers_;
    YOLOLoss criterion_;
    float gradient_clip_ { dl::kDefaultGradientClip };
};
