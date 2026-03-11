#include "DeepLearnLib/tensor.hpp"
#include <numeric>

namespace dl
{
Tensor::Tensor(std::vector<int> shape, auto device)
    : m_shape(std::move(shape))
    , m_device(device)
{
    compute_strides();
    m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<int>());

    if (m_device == Device::GPU)
    {
        if (cudaGetDeviceCount(nullptr) == 0)
        {
            throw std::runtime_error("No CUDA-capable devices found");
        }
        float* gpu_data;
        cudaMalloc(&gpu_data, m_size * sizeof(float));
        m_data = std::shared_ptr<float>(gpu_data, CudaDeleter());
    }
    else
    {
        float* cpu_data = new float[m_size]();
        m_data = std::shared_ptr<float>(cpu_data, CpuDeleter());
    }
}

Tensor::Tensor(std::vector<int> shape, std::vector<int> strides, std::shared_ptr<float> data, Device device)
    : m_shape(std::move(shape))
    , m_strides(std::move(strides))
    , m_data(std::move(data))
    , m_device(device)
{
    m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<int>());
}
} // namespace dl
