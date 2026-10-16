#include "DeepLearnLib/core/Precision.hpp"

namespace dl
{

auto dtype_name(Dtype dtype) -> const char*
{
    switch (dtype)
    {
    case Dtype::Float32:
        return "fp32";
    }

    return "fp32";
}

}
