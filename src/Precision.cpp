#include "DeepLearnLib/Precision.hpp"

namespace dl
{

// Short dtype name for logs. The only implemented value is fp32, including the default branch.
auto dtype_name(Dtype dtype) -> const char*
{
    switch (dtype)
    {
    case Dtype::Float32:
        return "fp32";
    }
    // The switch names no other dtype, so this return is fp32 as well.
    return "fp32";
}

} // namespace dl
