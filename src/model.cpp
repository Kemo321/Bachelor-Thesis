#include "DeepLearnLib/model.hpp"

YOLOv1Impl::YOLOv1Impl()
{
    using namespace torch::nn;

    backbone = register_module("backbone", Sequential(
        Conv2d(Conv2dOptions(3, 64, 7).stride(2).padding(3)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        Conv2d(Conv2dOptions(64, 192, 3).padding(1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        Conv2d(Conv2dOptions(192, 128, 1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(128, 256, 3).padding(1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(256, 256, 1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        MaxPool2d(MaxPool2dOptions(2).stride(2)),
        
        Conv2d(Conv2dOptions(512, 256, 1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(512, 256, 1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(512, 512, 1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(512, 1024, 3).padding(1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        Conv2d(Conv2dOptions(1024, 512, 1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(512, 1024, 3).padding(1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(1024, 512, 1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(512, 1024, 3).padding(1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(1024, 1024, 3).padding(1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(1024, 1024, 3).stride(2).padding(1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(1024, 1024, 3).padding(1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), 
        Conv2d(Conv2dOptions(1024, 1024, 3).padding(1)), 
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1))
    ));

    head = register_module("head", Sequential(
        Linear(7 * 7 * 1024, 4096), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)), Dropout(DropoutOptions(0.5)), Linear(4096, 7 * 7 * 30)
    ));
}

torch::Tensor YOLOv1Impl::forward(torch::Tensor x)
{
    x = backbone->forward(x);
    x = x.view({ x.size(0), -1 });
    x = head->forward(x);
    return x.view({ -1, 7, 7, 30 });
}

torch::Tensor compute_yolo_loss(const torch::Tensor& pred, const torch::Tensor& target)
{
    const float lambda_coord = 5.0F;
    const float lambda_noobj = 0.5F;

    // Maski dla każdej z dwóch ramek (box0 i box1)
    auto obj_mask0   = target.slice(3, 4, 5)  > 0.5F;   // box 0 ma obiekt
    auto obj_mask1   = target.slice(3, 9, 10) > 0.5F;   // box 1 ma obiekt
    auto noobj_mask0 = target.slice(3, 4, 5)  <= 0.5F;
    auto noobj_mask1 = target.slice(3, 9, 10) <= 0.5F;

    // Komórka siatki zawiera jakikolwiek obiekt (do straty klas)
    auto obj_mask_cell = obj_mask0 | obj_mask1;

    // Pomocnicza funkcja – unika NaN przy pustych maskach (gdy np. box1 nie ma żadnych obiektów)
    auto safe_mse = [&](const torch::Tensor& p, const torch::Tensor& t, const torch::Tensor& m) -> torch::Tensor {
        auto sel_p = p.masked_select(m);
        if (sel_p.numel() == 0) {
            return torch::tensor(0.0F, p.options());
        }
        auto sel_t = t.masked_select(m);
        return torch::mse_loss(sel_p, sel_t);
    };

    // ==================== STRATA KOORDYNATÓW (tylko dla odpowiedzialnej ramki) ====================
    // Box 0
    auto pred_xy0 = pred.slice(3, 0, 2);
    auto target_xy0 = target.slice(3, 0, 2);
    auto pred_wh0 = torch::sqrt(torch::clamp(pred.slice(3, 2, 4), 1e-6F));
    auto target_wh0 = torch::sqrt(torch::clamp(target.slice(3, 2, 4), 1e-6F));

    auto loss_xy0 = safe_mse(pred_xy0, target_xy0, obj_mask0.expand({-1, -1, -1, 2}));
    auto loss_wh0 = safe_mse(pred_wh0, target_wh0, obj_mask0.expand({-1, -1, -1, 2}));

    // Box 1
    auto pred_xy1 = pred.slice(3, 5, 7);
    auto target_xy1 = target.slice(3, 5, 7);
    auto pred_wh1 = torch::sqrt(torch::clamp(pred.slice(3, 7, 9), 1e-6F));
    auto target_wh1 = torch::sqrt(torch::clamp(target.slice(3, 7, 9), 1e-6F));

    auto loss_xy1 = safe_mse(pred_xy1, target_xy1, obj_mask1.expand({-1, -1, -1, 2}));
    auto loss_wh1 = safe_mse(pred_wh1, target_wh1, obj_mask1.expand({-1, -1, -1, 2}));

    auto loss_coord = loss_xy0 + loss_xy1 + loss_wh0 + loss_wh1;

    // ==================== STRATA CONFIDENCE ====================
    auto pred_conf0 = pred.slice(3, 4, 5);
    auto target_conf0 = target.slice(3, 4, 5);
    auto pred_conf1 = pred.slice(3, 9, 10);
    auto target_conf1 = target.slice(3, 9, 10);

    auto loss_conf_obj0   = safe_mse(pred_conf0, target_conf0, obj_mask0);
    auto loss_conf_obj1   = safe_mse(pred_conf1, target_conf1, obj_mask1);
    auto loss_conf_obj    = loss_conf_obj0 + loss_conf_obj1;

    auto loss_conf_noobj0 = safe_mse(pred_conf0, target_conf0, noobj_mask0);
    auto loss_conf_noobj1 = safe_mse(pred_conf1, target_conf1, noobj_mask1);
    auto loss_conf_noobj  = loss_conf_noobj0 + loss_conf_noobj1;

    // ==================== STRATA KLAS ====================
    auto pred_class = pred.slice(3, 10, 30);
    auto target_class = target.slice(3, 10, 30);
    auto loss_class = safe_mse(
        pred_class,
        target_class,
        obj_mask_cell.expand({-1, -1, -1, 20})
    );

    // ==================== ŁĄCZNA STRATA ====================
    return lambda_coord * loss_coord +
           loss_conf_obj +
           lambda_noobj * loss_conf_noobj +
           loss_class;
}