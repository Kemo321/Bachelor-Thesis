#include "DeepLearnLib/CSVLoader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

struct ParsedCsv
{
    std::vector<float> features;
    std::vector<float> targets;
    int batch = 0;
    int feature_columns = 0;
    int target_columns = 0;
};

auto parse_csv(const std::string& csv_path, int target_columns, bool skip_header) -> ParsedCsv
{
    if (target_columns <= 0)
    {
        throw std::runtime_error("CSVLoader target_columns must be positive");
    }

    std::ifstream stream(csv_path);
    if (!stream)
    {
        throw std::runtime_error("CSVLoader could not open file: " + csv_path);
    }

    std::string line;
    if (skip_header && !std::getline(stream, line))
    {
        throw std::runtime_error("CSVLoader expected a header line: " + csv_path);
    }

    std::vector<std::vector<float>> rows;
    int width = -1;
    while (std::getline(stream, line))
    {
        if (line.empty() || line == "\r")
        {
            continue;
        }
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        std::vector<float> values;
        std::stringstream row_stream(line);
        std::string cell;
        while (std::getline(row_stream, cell, ','))
        {
            if (cell.empty())
            {
                throw std::runtime_error("CSVLoader found an empty cell in " + csv_path);
            }
            try
            {
                values.push_back(std::stof(cell));
            }
            catch (const std::exception&)
            {
                throw std::runtime_error("CSVLoader expected a float in " + csv_path + ", got: " + cell);
            }
        }
        if (values.empty())
        {
            continue;
        }
        if (width < 0)
        {
            width = static_cast<int>(values.size());
        }
        else if (static_cast<int>(values.size()) != width)
        {
            throw std::runtime_error("CSVLoader requires a rectangular CSV: " + csv_path);
        }
        rows.push_back(std::move(values));
    }

    if (rows.empty())
    {
        throw std::runtime_error("CSVLoader found no data rows in " + csv_path);
    }
    if (width <= target_columns)
    {
        throw std::runtime_error("CSVLoader needs more columns than target_columns");
    }

    ParsedCsv parsed;
    parsed.batch = static_cast<int>(rows.size());
    parsed.feature_columns = width - target_columns;
    parsed.target_columns = target_columns;
    parsed.features.resize(static_cast<std::size_t>(parsed.batch) * static_cast<std::size_t>(parsed.feature_columns));
    parsed.targets.resize(static_cast<std::size_t>(parsed.batch) * static_cast<std::size_t>(parsed.target_columns));
    for (int row = 0; row < parsed.batch; ++row)
    {
        for (int col = 0; col < parsed.feature_columns; ++col)
        {
            parsed.features[(static_cast<std::size_t>(row) * static_cast<std::size_t>(parsed.feature_columns))
                + static_cast<std::size_t>(col)]
                = rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
        }
        for (int col = 0; col < parsed.target_columns; ++col)
        {
            parsed.targets[(static_cast<std::size_t>(row) * static_cast<std::size_t>(parsed.target_columns))
                + static_cast<std::size_t>(col)]
                = rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(parsed.feature_columns + col)];
        }
    }
    return parsed;
}

} // namespace

CSVLoader::CSVLoader(dl::Tensor features, dl::Tensor targets)
    : features_(std::move(features))
    , targets_(std::move(targets))
{
}

auto CSVLoader::from_parsed(const std::string& csv_path, int target_columns, bool skip_header) -> CSVLoader
{
    const ParsedCsv parsed = parse_csv(csv_path, target_columns, skip_header);
    return CSVLoader(dl::Tensor::from_host({ parsed.batch, parsed.feature_columns }, parsed.features, dl::Device::GPU),
        dl::Tensor::from_host({ parsed.batch, parsed.target_columns }, parsed.targets, dl::Device::GPU));
}

CSVLoader::CSVLoader(std::string csv_path, int target_columns, bool skip_header)
    : CSVLoader(from_parsed(csv_path, target_columns, skip_header))
{
}

auto CSVLoader::features() const -> const dl::Tensor&
{
    return features_;
}

auto CSVLoader::targets() const -> const dl::Tensor&
{
    return targets_;
}

auto CSVLoader::size() const -> std::size_t
{
    return static_cast<std::size_t>(features_.get_shape()[0]);
}
