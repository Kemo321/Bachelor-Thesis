#include "DeepLearnLib/loaders/Ecg200Loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

struct ParsedRow
{
    double label = 0.0;
    std::vector<float> values;
};

auto trim(std::string text) -> std::string
{
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0)
    {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0)
    {
        text.pop_back();
    }
    return text;
}

auto split_char(const std::string& text, char separator) -> std::vector<std::string>
{
    std::vector<std::string> parts;
    std::string item;
    std::stringstream stream(text);
    while (std::getline(stream, item, separator))
    {
        parts.push_back(trim(item));
    }
    return parts;
}

auto parse_floats(const std::string& text) -> std::vector<float>
{
    std::vector<float> values;
    std::stringstream stream(text);
    double value = 0.0;
    while (stream >> value)
    {
        values.push_back(static_cast<float>(value));
    }
    return values;
}

auto parse_row(const std::string& raw) -> std::optional<ParsedRow>
{
    const std::string line = trim(raw);
    if (line.empty() || line.front() == '#' || line.front() == '@')
    {
        return std::nullopt;
    }

    ParsedRow row;
    const auto colon = line.find(':');
    if (colon != std::string::npos)
    {
        const std::string left = trim(line.substr(0, colon));
        const std::string right = trim(line.substr(colon + 1));
        const bool left_is_series = left.find(',') != std::string::npos;
        const bool right_is_series = right.find(',') != std::string::npos;
        std::string series = left;
        std::string label = right;
        if (right_is_series && !left_is_series)
        {
            series = right;
            label = left;
        }
        std::replace(series.begin(), series.end(), ',', ' ');
        row.values = parse_floats(series);
        row.label = std::stod(label);
    }
    else
    {
        std::string spaced = line;
        std::replace(spaced.begin(), spaced.end(), ',', ' ');
        const std::vector<float> tokens = parse_floats(spaced);
        if (tokens.size() < 2)
        {
            throw std::runtime_error("ECG200 row needs a label and at least one sample");
        }
        row.label = static_cast<double>(tokens.front());
        row.values.assign(tokens.begin() + 1, tokens.end());
    }

    if (row.values.empty())
    {
        throw std::runtime_error("ECG200 row has no samples");
    }
    return row;
}

auto label_name(double value) -> std::string
{
    if (std::floor(value) == value && std::abs(value) < 1.0e6)
    {
        return std::to_string(static_cast<int>(value));
    }
    return std::to_string(value);
}

}

auto load_ecg200_file(const std::filesystem::path& path) -> Ecg200Data
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("Failed to open ECG200 file: " + path.string());
    }

    std::vector<ParsedRow> rows;
    std::string line;
    while (std::getline(stream, line))
    {
        if (auto parsed = parse_row(line))
        {
            rows.push_back(std::move(*parsed));
        }
    }
    if (rows.empty())
    {
        throw std::runtime_error("ECG200 file has no series: " + path.string());
    }

    const int length = static_cast<int>(rows.front().values.size());
    for (const ParsedRow& row : rows)
    {
        if (static_cast<int>(row.values.size()) != length)
        {
            throw std::runtime_error("ECG200 series lengths differ in " + path.string());
        }
    }

    std::vector<double> label_values;
    for (const ParsedRow& row : rows)
    {
        if (std::find(label_values.begin(), label_values.end(), row.label) == label_values.end())
        {
            label_values.push_back(row.label);
        }
    }
    std::sort(label_values.begin(), label_values.end());
    if (label_values.size() < 2)
    {
        throw std::runtime_error("ECG200 file needs at least two classes: " + path.string());
    }

    Ecg200Data data;
    data.length = length;
    data.values.reserve(rows.size() * static_cast<std::size_t>(length));
    data.labels.reserve(rows.size());
    for (double label : label_values)
    {
        data.class_names.push_back(label_name(label));
    }
    for (const ParsedRow& row : rows)
    {
        const auto found = std::find(label_values.begin(), label_values.end(), row.label);
        data.labels.push_back(static_cast<int>(found - label_values.begin()));
        data.values.insert(data.values.end(), row.values.begin(), row.values.end());
    }
    return data;
}

auto find_ecg200_split(const std::filesystem::path& root, const std::string& which) -> std::filesystem::path
{
    const std::vector<std::string> names {
        "ECG200_" + which + ".txt",
        "ECG200_" + which + ".tsv",
        "ECG200_" + which + ".ts",
    };
    for (const std::string& name : names)
    {
        const std::filesystem::path direct = root / name;
        if (std::filesystem::exists(direct))
        {
            return direct;
        }
        const std::filesystem::path nested = root / "ECG200" / name;
        if (std::filesystem::exists(nested))
        {
            return nested;
        }
    }
    throw std::runtime_error("ECG200 " + which
        + " split not found under " + root.string()
        + ". Download https://www.timeseriesclassification.com/aeon-toolkit/ECG200.zip");
}
