#include "DeepLearnLib/tensor.hpp"
namespace dl {
    Tensor::Tensor(int size) : m_size(size) {}
    int Tensor::getSize() const { return m_size; }
}
