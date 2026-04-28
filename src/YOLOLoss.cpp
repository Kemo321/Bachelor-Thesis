#include "DeepLearnLib/YOLOLoss.hpp"

auto YOLOLoss::calculate_iou(const torch::Tensor& box1, const torch::Tensor& box2) -> torch::Tensor {
    auto b1_x1 = box1.select(-1, 0) - box1.select(-1, 2) / 2;
    auto b1_y1 = box1.select(-1, 1) - box1.select(-1, 3) / 2;
    auto b1_x2 = box1.select(-1, 0) + box1.select(-1, 2) / 2;
    auto b1_y2 = box1.select(-1, 1) + box1.select(-1, 3) / 2;

    auto b2_x1 = box2.select(-1, 0) - box2.select(-1, 2) / 2;
    auto b2_y1 = box2.select(-1, 1) - box2.select(-1, 3) / 2;
    auto b2_x2 = box2.select(-1, 0) + box2.select(-1, 2) / 2;
    auto b2_y2 = box2.select(-1, 1) + box2.select(-1, 3) / 2;

    auto inter_x1 = torch::max(b1_x1, b2_x1);
    auto inter_y1 = torch::max(b1_y1, b2_y1);
    auto inter_x2 = torch::min(b1_x2, b2_x2);
    auto inter_y2 = torch::min(b1_y2, b2_y2);

    auto inter_area = torch::clamp(inter_x2 - inter_x1, 0) * torch::clamp(inter_y2 - inter_y1, 0);
    auto area1 = torch::clamp(box1.select(-1, 2) * box1.select(-1, 3), 1e-6F);
    auto area2 = torch::clamp(box2.select(-1, 2) * box2.select(-1, 3), 1e-6F);

    return inter_area / (area1 + area2 - inter_area + 1e-6F);
}

auto YOLOLoss::loss(const torch::Tensor& target, const torch::Tensor& prediction) -> torch::Tensor {
    auto pred = (prediction.dim() == 2) ? prediction.view({-1, 7, 7, 30}) : prediction;
    auto tgt  = (target.dim() == 2)     ? target.view({-1, 7, 7, 30})     : target;

    int64_t batch_size = pred.size(0);

    constexpr float lambda_coord = 5.0F;
    constexpr float lambda_noobj = 0.5F;
    constexpr float eps = 1e-6F;
    constexpr float safe_eps = 1e-8F;

    auto grid = torch::meshgrid({torch::arange(7, pred.options()), torch::arange(7, pred.options())}, "ij");
    auto grid_y = grid[0].view({1, 7, 7, 1}).expand({batch_size, 7, 7, 1});
    auto grid_x = grid[1].view({1, 7, 7, 1}).expand({batch_size, 7, 7, 1});

    auto p_box1 = pred.slice(3, 0, 4).clone();
    p_box1.select(3, 0) = (p_box1.select(3, 0).unsqueeze(-1) + grid_x).squeeze(-1) / 7.0F;
    p_box1.select(3, 1) = (p_box1.select(3, 1).unsqueeze(-1) + grid_y).squeeze(-1) / 7.0F;

    auto p_box2 = pred.slice(3, 5, 9).clone();
    p_box2.select(3, 0) = (p_box2.select(3, 0).unsqueeze(-1) + grid_x).squeeze(-1) / 7.0F;
    p_box2.select(3, 1) = (p_box2.select(3, 1).unsqueeze(-1) + grid_y).squeeze(-1) / 7.0F;

    auto t_box = tgt.slice(3, 0, 4).clone();
    t_box.select(3, 0) = (t_box.select(3, 0).unsqueeze(-1) + grid_x).squeeze(-1) / 7.0F;
    t_box.select(3, 1) = (t_box.select(3, 1).unsqueeze(-1) + grid_y).squeeze(-1) / 7.0F;

    auto iou1 = calculate_iou(p_box1, t_box);
    auto iou2 = calculate_iou(p_box2, t_box);

    auto obj_mask = tgt.slice(3, 4, 5).squeeze(-1);
    auto box2_better = (iou2 > iou1).to(torch::kFloat32);
    auto resp_b1 = (1.0F - box2_better) * obj_mask;
    auto resp_b2 = box2_better * obj_mask;

    auto l_coord = lambda_coord * (
        (pred.slice(3, 0, 2) - tgt.slice(3, 0, 2)).pow(2).sum({3}).mul(resp_b1).sum() +
        (pred.slice(3, 5, 7) - tgt.slice(3, 0, 2)).pow(2).sum({3}).mul(resp_b2).sum() +
        (torch::sqrt(torch::clamp(pred.slice(3, 2, 4), eps)) - torch::sqrt(torch::clamp(tgt.slice(3, 2, 4), eps)))
            .pow(2).sum({3}).mul(resp_b1).sum() +
        (torch::sqrt(torch::clamp(pred.slice(3, 7, 9), eps)) - torch::sqrt(torch::clamp(tgt.slice(3, 2, 4), eps)))
            .pow(2).sum({3}).mul(resp_b2).sum()
    );

    auto noobj_mask_b1 = (1.0F - resp_b1);
    auto noobj_mask_b2 = (1.0F - resp_b2);

    auto l_conf = (
        (pred.slice(3, 4, 5).squeeze(-1) - iou1).pow(2).mul(resp_b1).sum() +
        (pred.slice(3, 9, 10).squeeze(-1) - iou2).pow(2).mul(resp_b2).sum() +
        lambda_noobj * pred.slice(3, 4, 5).squeeze(-1).pow(2).mul(noobj_mask_b1).sum() +
        lambda_noobj * pred.slice(3, 9, 10).squeeze(-1).pow(2).mul(noobj_mask_b2).sum()
    );

    auto l_class = (pred.slice(3, 10, 30) - tgt.slice(3, 10, 30)).pow(2).sum({3}).mul(obj_mask).sum();

    return (l_coord + l_conf + l_class) / static_cast<float>(batch_size);
}

auto YOLOLoss::loss_derivative(const torch::Tensor& target, const torch::Tensor& prediction) -> torch::Tensor {
    auto pred = (prediction.dim() == 2) ? prediction.view({-1, 7, 7, 30}) : prediction;
    auto tgt  = (target.dim() == 2)     ? target.view({-1, 7, 7, 30})     : target;

    auto grad = torch::zeros_like(pred);

    int64_t batch_size = pred.size(0);
    constexpr float lambda_coord = 5.0F;
    constexpr float lambda_noobj = 0.5F;
    constexpr float eps = 1e-6F;
    constexpr float safe_eps = 1e-8F;

    auto grid_x = torch::arange(7, pred.options()).view({1, 1, 7, 1}).expand({batch_size, 7, 7, 1});
    auto grid_y = torch::arange(7, pred.options()).view({1, 7, 1, 1}).expand({batch_size, 7, 7, 1});

    auto p_box1 = pred.slice(3, 0, 4).clone();
    p_box1.select(3, 0) = (p_box1.select(3, 0).unsqueeze(-1) + grid_x).squeeze(-1) / 7.0F;
    p_box1.select(3, 1) = (p_box1.select(3, 1).unsqueeze(-1) + grid_y).squeeze(-1) / 7.0F;

    auto p_box2 = pred.slice(3, 5, 9).clone();
    p_box2.select(3, 0) = (p_box2.select(3, 0).unsqueeze(-1) + grid_x).squeeze(-1) / 7.0F;
    p_box2.select(3, 1) = (p_box2.select(3, 1).unsqueeze(-1) + grid_y).squeeze(-1) / 7.0F;

    auto t_box = tgt.slice(3, 0, 4).clone();
    t_box.select(3, 0) = (t_box.select(3, 0).unsqueeze(-1) + grid_x).squeeze(-1) / 7.0F;
    t_box.select(3, 1) = (t_box.select(3, 1).unsqueeze(-1) + grid_y).squeeze(-1) / 7.0F;

    auto iou1 = calculate_iou(p_box1, t_box);
    auto iou2 = calculate_iou(p_box2, t_box);

    auto obj_mask = tgt.slice(3, 4, 5).squeeze(-1);
    auto box2_better = (iou2 > iou1).to(torch::kFloat32);
    auto resp_b1 = (1.0F - box2_better) * obj_mask;
    auto resp_b2 = box2_better * obj_mask;

    auto noobj_mask_b1 = (1.0F - resp_b1);
    auto noobj_mask_b2 = (1.0F - resp_b2);

    grad.slice(3, 0, 2) = 2.0F * lambda_coord * (pred.slice(3, 0, 2) - tgt.slice(3, 0, 2)) * resp_b1.unsqueeze(-1);
    grad.slice(3, 5, 7) = 2.0F * lambda_coord * (pred.slice(3, 5, 7) - tgt.slice(3, 0, 2)) * resp_b2.unsqueeze(-1);

    auto p1_wh_raw = pred.slice(3, 2, 4);
    auto mask_p1 = (p1_wh_raw > safe_eps).to(torch::kFloat32);

    auto p2_wh_raw = pred.slice(3, 7, 9);
    auto mask_p2 = (p2_wh_raw > safe_eps).to(torch::kFloat32);

    auto sqrt_p1 = torch::sqrt(torch::clamp(p1_wh_raw, safe_eps));
    auto sqrt_p2 = torch::sqrt(torch::clamp(p2_wh_raw, safe_eps));
    auto sqrt_t  = torch::sqrt(torch::clamp(tgt.slice(3, 2, 4), safe_eps));

    grad.slice(3, 2, 4) = lambda_coord * (sqrt_p1 - sqrt_t) / (sqrt_p1 + safe_eps) * mask_p1 * resp_b1.unsqueeze(-1);
    grad.slice(3, 7, 9) = lambda_coord * (sqrt_p2 - sqrt_t) / (sqrt_p2 + safe_eps) * mask_p2 * resp_b2.unsqueeze(-1);

    grad.slice(3, 4, 5) = 2.0F * (pred.slice(3, 4, 5).squeeze(-1) - iou1).unsqueeze(-1) * resp_b1.unsqueeze(-1);
    grad.slice(3, 4, 5) += 2.0F * lambda_noobj * pred.slice(3, 4, 5) * noobj_mask_b1.unsqueeze(-1);

    grad.slice(3, 9, 10) = 2.0F * (pred.slice(3, 9, 10).squeeze(-1) - iou2).unsqueeze(-1) * resp_b2.unsqueeze(-1);
    grad.slice(3, 9, 10) += 2.0F * lambda_noobj * pred.slice(3, 9, 10) * noobj_mask_b2.unsqueeze(-1);

    grad.slice(3, 10, 30) = 2.0F * (pred.slice(3, 10, 30) - tgt.slice(3, 10, 30)) * obj_mask.unsqueeze(-1);

    auto final_grad = grad / static_cast<float>(batch_size);

    return (prediction.dim() == 2) ? final_grad.view({batch_size, -1}) : final_grad;
}