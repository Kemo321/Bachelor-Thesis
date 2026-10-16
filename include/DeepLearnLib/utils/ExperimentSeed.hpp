#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace dl
{

inline constexpr unsigned kExperimentSeed = 42U;

inline constexpr unsigned long long kConvWeightSeed = 0xC0FFEEULL;
inline constexpr unsigned long long kConvBiasSeed = 0xBADC0DEULL;
inline constexpr unsigned long long kLinearWeightSeed = 0xF00DULL;
inline constexpr unsigned long long kLinearBiasSeed = 0xBEEFULL;

#if defined(__CUDACC__)
#define DL_PAIRED_UNIFORM_ATTR __host__ __device__
#else
#define DL_PAIRED_UNIFORM_ATTR
#endif

DL_PAIRED_UNIFORM_ATTR inline auto paired_uniform(int index, float low, float high, unsigned long long seed) -> float
{
    unsigned long long hash = seed + (static_cast<unsigned long long>(index) + 1ULL) * 0x9E3779B97F4A7C15ULL;
    hash ^= hash >> 30U;
    hash *= 0xBF58476D1CE4E5B9ULL;
    hash ^= hash >> 27U;
    hash *= 0x94D049BB133111EBULL;
    hash ^= hash >> 31U;
    const float unit = static_cast<float>(hash & 0xFFFFFFULL) / static_cast<float>(0x1000000ULL);
    return low + ((high - low) * unit);
}

#undef DL_PAIRED_UNIFORM_ATTR

inline auto uniform_bound(float fan_in) -> float
{
    const float denom = fan_in > 1.0e-7F ? fan_in : 1.0e-7F;
    return 1.0F / std::sqrt(denom);
}

inline auto next_bounded(std::mt19937& rng, std::uint32_t bound) -> std::uint32_t
{
    if (bound <= 1U)
    {
        return 0U;
    }
    const std::uint32_t limit = (std::numeric_limits<std::uint32_t>::max() / bound) * bound;
    std::uint32_t draw = 0U;
    do
    {
        draw = static_cast<std::uint32_t>(rng());
    } while (draw >= limit);
    return draw % bound;
}

template <typename T>
auto portable_shuffle(std::vector<T>& values, std::mt19937& rng) -> void
{
    for (std::size_t index = values.size(); index > 1U; --index)
    {
        const std::size_t pick = static_cast<std::size_t>(next_bounded(rng, static_cast<std::uint32_t>(index)));
        std::swap(values[index - 1U], values[pick]);
    }
}

}
