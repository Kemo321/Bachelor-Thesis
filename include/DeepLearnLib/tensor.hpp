#pragma once

namespace dl
{
class Tensor
{
public:
    explicit Tensor(int size);

private:
    int m_size = 0;
};
}
