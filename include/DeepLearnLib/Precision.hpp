#pragma once

#include <cstddef>
#include <string>

namespace dl
{

enum class Dtype
{
    Float32,
    Float16
};

[[nodiscard]] constexpr auto element_size(Dtype dtype) -> std::size_t
{
    return dtype == Dtype::Float16 ? 2U : 4U;
}

[[nodiscard]] auto dtype_name(Dtype dtype) -> const char*;

/**
 * Process-wide mixed-precision policy.
 *
 * Default is FP32. Enabling mixed precision allocates/computes in FP16 (Tensor
 * Cores) and applies static loss scaling. Call this before constructing models.
 */
auto set_mixed_precision(bool enabled, float loss_scale = 1024.0F) -> void;
[[nodiscard]] auto mixed_precision_enabled() -> bool;
[[nodiscard]] auto compute_dtype() -> Dtype;
[[nodiscard]] auto loss_scale() -> float;

/** Gradient-clip bound in the same units as scaled backward gradients. */
[[nodiscard]] inline auto scaled_gradient_clip(float bound) -> float
{
    return bound * loss_scale();
}

/**
 * Parse pipeline JSON keys `mixed_precision`, `precision`, and `loss_scale`.
 *
 * FP32 is used when mixed_precision is false, precision is "fp32"/"float", or
 * the keys are omitted. FP16 requires mixed_precision true and/or precision
 * "fp16"/"half".
 */
auto configure_precision(bool mixed_precision, const std::string& precision, float loss_scale_value) -> void;

class MixedPrecisionGuard
{
public:
    explicit MixedPrecisionGuard(bool enabled, float scale = 1024.0F);
    ~MixedPrecisionGuard();

    MixedPrecisionGuard(const MixedPrecisionGuard&) = delete;
    auto operator=(const MixedPrecisionGuard&) -> MixedPrecisionGuard& = delete;
    MixedPrecisionGuard(MixedPrecisionGuard&&) = delete;
    auto operator=(MixedPrecisionGuard&&) -> MixedPrecisionGuard& = delete;

private:
    bool previous_enabled_ { false };
    float previous_scale_ { 1.0F };
};

} // namespace dl
