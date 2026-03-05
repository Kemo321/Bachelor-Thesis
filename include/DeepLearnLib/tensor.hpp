#pragma once

namespace dl
{
class Tensor
{
public:
    explicit Tensor(int size);
    int getSize() const;

private:
    int m_size = 0;
};
}
