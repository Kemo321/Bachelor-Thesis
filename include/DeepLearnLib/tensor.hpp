#pragma once
namespace dl
{
class Tensor
{
public:
    Tensor(int size);
    int getSize() const;

private:
    int m_size;
};
}
