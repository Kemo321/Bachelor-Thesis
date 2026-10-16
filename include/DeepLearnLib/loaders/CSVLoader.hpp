#pragma once

#include "DeepLearnLib/core/Tensor.hpp"

#include <cstddef>
#include <string>

class CSVLoader
{
public:
    explicit CSVLoader(std::string csv_path, int target_columns = 1, bool skip_header = false);

    [[nodiscard]] auto features() const -> const dl::Tensor&;
    [[nodiscard]] auto targets() const -> const dl::Tensor&;
    [[nodiscard]] auto size() const -> std::size_t;

private:
    static auto from_parsed(const std::string& csv_path, int target_columns, bool skip_header) -> CSVLoader;
    CSVLoader(dl::Tensor features, dl::Tensor targets);

    dl::Tensor features_;
    dl::Tensor targets_;
};
