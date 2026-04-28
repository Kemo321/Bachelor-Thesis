#include "DeepLearnLib/TorchYOLO.hpp"

YOLOv1Impl::YOLOv1Impl()
{
    using namespace torch::nn;

    backbone = register_module("backbone", Sequential(
        Conv2d(Conv2dOptions(3, 64, 7).stride(2).padding(3)),
        BatchNorm2d(64),
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        Conv2d(Conv2dOptions(64, 192, 3).padding(1)),
        BatchNorm2d(192),
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        Conv2d(Conv2dOptions(192, 128, 1)), BatchNorm2d(128), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(128, 256, 3).padding(1)), BatchNorm2d(256), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(256, 256, 1)), BatchNorm2d(256), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        Conv2d(Conv2dOptions(512, 256, 1)), BatchNorm2d(256), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(512, 256, 1)), BatchNorm2d(256), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(512, 256, 1)), BatchNorm2d(256), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(512, 256, 1)), BatchNorm2d(256), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(512, 512, 1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(512, 1024, 3).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        Conv2d(Conv2dOptions(1024, 512, 1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(512, 1024, 3).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(1024, 512, 1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(512, 1024, 3).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),

        Conv2d(Conv2dOptions(1024, 1024, 3).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(1024, 1024, 3).stride(2).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(1024, 1024, 3).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Conv2d(Conv2dOptions(1024, 1024, 3).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1))
    ));

    head = register_module("head", Sequential(
        Linear(1024 * 7 * 7, 4096),
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1)),
        Dropout(0.5),
        Linear(4096, 7 * 7 * 30)
    ));
}

auto YOLOv1Impl::forward(torch::Tensor input_tensor) -> torch::Tensor
{
    auto x = backbone->forward(input_tensor);
    x = x.view({ x.size(0), -1 });
    x = head->forward(x);
    return x.view({ -1, 7, 7, 30 });
}