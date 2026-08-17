#include "DeepLearnLib/Precision.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace dl
{
namespace
{

    struct PrecisionState
    {
        bool enabled { false };
        float scale { 1.0F };
    };

    auto state() -> PrecisionState&
    {
        static PrecisionState current;
        return current;
    }

    auto normalize_precision(const std::string& precision) -> std::string
    {
        std::string lower = precision;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char character)
            { return static_cast<char>(std::tolower(character)); });
        return lower;
    }

} // namespace

auto dtype_name(Dtype dtype) -> const char*
{
    return dtype == Dtype::Float16 ? "fp16" : "fp32";
}

auto set_mixed_precision(bool enabled, float loss_scale_value) -> void
{
    if (loss_scale_value <= 0.0F)
    {
        throw std::runtime_error("loss_scale must be positive");
    }

    state().enabled = enabled;
    state().scale = enabled ? loss_scale_value : 1.0F;

#if DEEPLEARNLIB_ENABLE_CUDA
    const cublasMath_t math_mode = enabled ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    CHECK_CUBLAS(cublasSetMathMode(get_cublas_handle(), math_mode));
#endif

    if (enabled)
    {
        log_info_message(std::string("Mixed precision enabled (fp16 Tensor Cores, loss_scale=")
            + std::to_string(state().scale) + ")");
    }
    else
    {
        log_info_message("Mixed precision disabled; using fp32");
    }
}

auto mixed_precision_enabled() -> bool
{
    return state().enabled;
}

auto compute_dtype() -> Dtype
{
    return state().enabled ? Dtype::Float16 : Dtype::Float32;
}

auto loss_scale() -> float
{
    return state().scale;
}

auto configure_precision(bool mixed_precision, const std::string& precision, float loss_scale_value) -> void
{
    const std::string normalized = normalize_precision(precision);
    const bool explicit_fp32 = normalized == "fp32" || normalized == "float32" || normalized == "float";
    const bool explicit_fp16 = normalized == "fp16" || normalized == "float16" || normalized == "half";
    const bool enable = !explicit_fp32 && (mixed_precision || explicit_fp16);
    const float scale = enable ? loss_scale_value : 1.0F;
    set_mixed_precision(enable, scale);
}

MixedPrecisionGuard::MixedPrecisionGuard(bool enabled, float scale)
    : previous_enabled_(mixed_precision_enabled())
    , previous_scale_(loss_scale())
{
    set_mixed_precision(enabled, scale);
}

MixedPrecisionGuard::~MixedPrecisionGuard()
{
    set_mixed_precision(previous_enabled_, previous_scale_);
}

} // namespace dl
