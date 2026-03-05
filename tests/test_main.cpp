#include <iostream>
#include "DeepLearnLib/tensor.hpp"

int main() {
    dl::Tensor t(10);
    if (t.getSize() == 10) {
        std::cout << "Test Passed!" << std::endl;
        return 0;
    }
    std::cout << "Test Failed!" << std::endl;
    return 1;
}
