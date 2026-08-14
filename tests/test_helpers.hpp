#pragma once

#include "DeepLearnLib/Tensor.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace dllib_test
{
constexpr float kEpsilon = 1e-4F;
constexpr float kLooseEpsilon = 1e-3F;
constexpr float kTensorEpsilon = 1e-5F;

inline auto has_cuda_device() -> bool
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        return false;
    }
    return count > 0;
}

inline auto synchronize_device() -> void
{
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

inline auto expect_near_vector(const std::vector<float>& actual, const std::vector<float>& expected,
    float epsilon = kEpsilon) -> void
{
    EXPECT_EQ(actual.size(), expected.size());
    if (actual.size() != expected.size())
    {
        return;
    }
    for (size_t index = 0; index < actual.size(); ++index)
    {
        EXPECT_NEAR(actual[index], expected[index], epsilon) << "mismatch at index " << index;
    }
}

inline auto expect_all_finite(const std::vector<float>& values) -> void
{
    for (size_t index = 0; index < values.size(); ++index)
    {
        EXPECT_TRUE(std::isfinite(values[index])) << "non-finite value at index " << index;
    }
}

inline auto set_named_parameter(std::map<std::string, dl::Tensor>& params, const std::string& name,
    const std::vector<int>& shape, const std::vector<float>& host) -> void
{
    params.emplace(name, dl::Tensor::from_host(shape, host, dl::Device::GPU));
}

class GpuTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!has_cuda_device())
        {
            GTEST_SKIP() << "No CUDA-capable device available";
        }
    }
};
} // namespace dllib_test
