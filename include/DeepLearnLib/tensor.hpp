#pragma once

namespace dl
{
class Tensor
{
public:
    explicit Tensor(int size);
    auto get_size() const -> int;

private:
    int m_size = 0;
};
}
