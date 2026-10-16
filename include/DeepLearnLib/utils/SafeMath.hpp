#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define DL_HOST_DEVICE __host__ __device__
#else
#define DL_HOST_DEVICE
#endif

namespace dl
{

constexpr float kSafeEps = 1e-7F;

constexpr float kDefaultGradientClip = 0.0F;

DL_HOST_DEVICE inline auto safe_sqrt(float value) -> float
{
    return sqrtf(fmaxf(value, kSafeEps));
}

DL_HOST_DEVICE inline auto safe_div(float numerator, float denominator) -> float
{
    return numerator / (denominator + kSafeEps);
}

DL_HOST_DEVICE inline auto guarded_div(float numerator, float denominator) -> float
{
    return numerator / fmaxf(denominator, kSafeEps);
}

DL_HOST_DEVICE inline auto safe_inv(float denominator) -> float
{
    return 1.0F / fmaxf(denominator, kSafeEps);
}

DL_HOST_DEVICE inline auto clamp_unit(float probability) -> float
{
    return fminf(fmaxf(probability, kSafeEps), 1.0F);
}

DL_HOST_DEVICE inline auto safe_log(float probability) -> float
{
    return logf(clamp_unit(probability));
}

}
