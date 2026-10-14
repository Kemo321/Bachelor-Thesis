#pragma once

#include <cstddef>

namespace dl
{

enum class Dtype
{
    Float32
};

[[nodiscard]] constexpr auto element_size(Dtype) -> std::size_t
{
    return sizeof(float);
}

[[nodiscard]] auto dtype_name(Dtype dtype) -> const char*;

} // namespace dl
