#pragma once

/**
 * Sample-weighted totals for one split.
 *
 * A batch loss from CrossEntropyLoss (and from torch::nn::CrossEntropyLoss) is the mean
 * over that batch. Weighting by the batch length keeps a short last batch from counting
 * as much as a full one. Accuracy is correct / seen, on both stacks.
 */
struct SplitMetrics
{
    double loss_mass = 0.0;
    int correct = 0;
    int seen = 0;
    int batches = 0;

    auto observe(float mean_loss, int n_correct, int count) -> void
    {
        if (count <= 0)
        {
            return;
        }
        loss_mass += static_cast<double>(mean_loss) * static_cast<double>(count);
        correct += n_correct;
        seen += count;
        ++batches;
    }

    [[nodiscard]] auto loss() const -> float
    {
        if (seen <= 0)
        {
            return 0.0F;
        }
        return static_cast<float>(loss_mass / static_cast<double>(seen));
    }

    [[nodiscard]] auto accuracy() const -> float
    {
        if (seen <= 0)
        {
            return 0.0F;
        }
        return static_cast<float>(correct) / static_cast<float>(seen);
    }
};
