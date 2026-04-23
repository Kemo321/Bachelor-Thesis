#include "DeepLearnLib/YOLOLoss.hpp"

using namespace torch::indexing;

auto YOLOLoss::loss(const torch::Tensor& y_true, const torch::Tensor& y_pred) -> float
{
    return torch::mse_loss(y_pred, y_true).item<float>();
}

auto YOLOLoss::loss_derivative(const torch::Tensor& target, const torch::Tensor& pred) -> torch::Tensor
{
    auto grad_pred = torch::zeros_like(pred);
    constexpr float lambda_coord = 5.0F;
    constexpr float lambda_noobj = 0.5F;
    constexpr float eps = 1e-6F;

    auto obj_mask = target.index({ Slice(), Slice(), Slice(), Slice(4, 5) }) > 0.5F;
    auto noobj_mask = target.index({ Slice(), Slice(), Slice(), Slice(4, 5) }) <= 0.5F;

    auto grad_xy = 2.0F * lambda_coord * (pred.index({ Slice(), Slice(), Slice(), Slice(0, 2) }) - target.index({ Slice(), Slice(), Slice(), Slice(0, 2) }));
    grad_pred.index_put_({ obj_mask.expand_as(grad_xy) }, grad_xy);

    auto pred_wh = pred.index({ Slice(), Slice(), Slice(), Slice(2, 4) });
    auto target_wh = target.index({ Slice(), Slice(), Slice(), Slice(2, 4) });
    auto grad_wh = lambda_coord * (torch::sqrt(torch::clamp(pred_wh, eps)) - torch::sqrt(torch::clamp(target_wh, eps))) / torch::sqrt(torch::clamp(pred_wh, eps));
    grad_pred.index_put_({ obj_mask.expand_as(grad_wh) }, grad_wh);

    auto grad_conf_obj = 2.0F * (pred.index({ Slice(), Slice(), Slice(), Slice(4, 5) }) - target.index({ Slice(), Slice(), Slice(), Slice(4, 5) }));
    grad_pred.index_put_({ obj_mask }, grad_conf_obj);

    auto grad_conf_noobj = 2.0F * lambda_noobj * (pred.index({ Slice(), Slice(), Slice(), Slice(4, 5) }) - target.index({ Slice(), Slice(), Slice(), Slice(4, 5) }));
    grad_pred.index_put_({ noobj_mask }, grad_conf_noobj);

    auto grad_class = 2.0F * (pred.index({ Slice(), Slice(), Slice(), Slice(10, 30) }) - target.index({ Slice(), Slice(), Slice(), Slice(10, 30) }));
    grad_pred.index_put_({ obj_mask.expand_as(grad_class) }, grad_class);

    return grad_pred;
}
