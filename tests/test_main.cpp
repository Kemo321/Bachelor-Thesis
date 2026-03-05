#include "DeepLearnLib/tensor.hpp"
#include <iostream>

int main()
{
    dl::Tensor t(10);
    if (t.get_size() == 10)
    {
        std::cout << "Test Passed!" << std::endl;
        return 0;
    }
    std::cout << "Test Failed!" << std::endl;
    return 1;
}
