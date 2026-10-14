#pragma once

#include "DeepLearnLib/utils/ExperimentSeed.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

struct IndexSplit
{
    std::vector<int> train;
    std::vector<int> test;
};

/** Deterministic hold-out. The same seed yields the same partition on both stacks. */
inline auto split_train_test(int count, float test_fraction, unsigned seed) -> IndexSplit
{
    if (count < 2)
    {
        throw std::runtime_error("A train/test split needs at least two rows");
    }
    if (!(test_fraction > 0.0F && test_fraction < 1.0F))
    {
        throw std::runtime_error("test_fraction must be in (0, 1)");
    }

    std::vector<int> order(static_cast<std::size_t>(count));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    dl::portable_shuffle(order, rng);

    int test_count = static_cast<int>(std::lround(static_cast<double>(test_fraction) * static_cast<double>(count)));
    test_count = std::clamp(test_count, 1, count - 1);

    IndexSplit split;
    split.train.assign(order.begin(), order.begin() + (count - test_count));
    split.test.assign(order.begin() + (count - test_count), order.end());
    return split;
}

struct HostRows
{
    const std::vector<float>* values = nullptr;
    const std::vector<float>* labels = nullptr;
    int rows = 0;
    int width = 0;
    std::vector<int> tail;
};

struct RowBatch
{
    std::vector<float> values;
    std::vector<float> labels;
    int count = 0;
    int width = 0;
};

inline auto batch_shape(int count, const std::vector<int>& tail) -> std::vector<int>
{
    std::vector<int> shape;
    shape.reserve(tail.size() + 1);
    shape.push_back(count);
    shape.insert(shape.end(), tail.begin(), tail.end());
    return shape;
}

inline auto batch_shape_i64(int count, const std::vector<int>& tail) -> std::vector<int64_t>
{
    std::vector<int64_t> shape;
    shape.reserve(tail.size() + 1);
    shape.push_back(count);
    for (const int dim : tail)
    {
        shape.push_back(dim);
    }
    return shape;
}

inline auto require_host_rows(const HostRows& table) -> void
{
    if (table.values == nullptr || table.labels == nullptr)
    {
        throw std::runtime_error("HostRows is missing values or labels");
    }
    if (table.rows < 1 || table.width < 1 || table.tail.empty())
    {
        throw std::runtime_error("HostRows needs a positive row count, width, and shape tail");
    }
    int product = 1;
    for (const int dim : table.tail)
    {
        if (dim < 1)
        {
            throw std::runtime_error("HostRows tail dimensions must be positive");
        }
        product *= dim;
    }
    if (product != table.width)
    {
        throw std::runtime_error("HostRows width does not match the shape tail");
    }
    if (table.values->size() != static_cast<std::size_t>(table.rows) * static_cast<std::size_t>(table.width)
        || table.labels->size() != static_cast<std::size_t>(table.rows))
    {
        throw std::runtime_error("HostRows storage does not match rows * width");
    }
}

inline auto gather_host_rows(const HostRows& table, const std::vector<int>& order, int start, int count) -> RowBatch
{
    RowBatch batch;
    batch.count = count;
    batch.width = table.width;
    batch.values.resize(static_cast<std::size_t>(count) * static_cast<std::size_t>(table.width));
    batch.labels.resize(static_cast<std::size_t>(count));
    for (int row = 0; row < count; ++row)
    {
        const int sample = order[static_cast<std::size_t>(start + row)];
        std::copy_n(table.values->data() + (static_cast<std::size_t>(sample) * static_cast<std::size_t>(table.width)),
            static_cast<std::size_t>(table.width),
            batch.values.data() + (static_cast<std::size_t>(row) * static_cast<std::size_t>(table.width)));
        batch.labels[static_cast<std::size_t>(row)] = (*table.labels)[static_cast<std::size_t>(sample)];
    }
    return batch;
}

inline auto identity_order(int count) -> std::vector<int>
{
    std::vector<int> order(static_cast<std::size_t>(count));
    std::iota(order.begin(), order.end(), 0);
    return order;
}
