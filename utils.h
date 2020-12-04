#ifndef _UTILS_H_
#define _UTILS_H_
#include <torch/torch.h>

torch::Tensor label2tensor(torch::Tensor label, torch::Tensor tensor) {
  for (int64_t i = 0; i < label.size(0); ++i) tensor[i].fill_(label[i]);
  return tensor;
}

void weights_init_normal(torch::nn::Module &module) {
  torch::NoGradGuard no_grad;
  if (auto *conv2d = module.as<torch::nn::Conv2d>()) {
    conv2d->weight.normal_(0.0, 0.02);
  } else if (auto *convTranspose2d = module.as<torch::nn::ConvTranspose2d>()) {
    convTranspose2d->weight.normal_(0.0, 0.02);
  }
}

#endif  //_UTILS_H_