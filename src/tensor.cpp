#include "DeepLearnLib/tensor.hpp"
namespace dl
{
Tensor::Tensor(int size)
    : m_size(size)
{
}
auto Tensor::get_size() const -> int
{
    return m_size;
}
}
