#ifndef _MODELS_H_
#define _MODELS_H_
#include <torch/torch.h>

struct TVLossImpl : torch::nn::Module {
  TVLossImpl(double TVLoss_weight = 1) : _TVLoss_weight(TVLoss_weight) {}

  torch::Tensor forward(torch::Tensor x) {
    auto x1 = x.clone().detach();
    x1 = x1.permute({3, 2, 1, 0});
    auto w_variance = torch::zeros(1);
    for (int64_t i = 1; i < x1.size(0); ++i)
      w_variance += torch::sum(torch::pow(x1[i] - x1[i - 1], 2));

    x1.transpose_(0, 1);
    auto h_variance = torch::zeros(1);
    for (int64_t i = 1; i < x1.size(0); ++i)
      h_variance += torch::sum(torch::pow(x1[i] - x1[i - 1], 2));
    auto loss = _TVLoss_weight * (w_variance + h_variance);
    return loss;
  }

  double _TVLoss_weight;
};
TORCH_MODULE(TVLoss);

struct IdentityImpl : torch::nn::Module {
  IdentityImpl() {}

  torch::Tensor forward(torch::Tensor x) { return x; }
};
TORCH_MODULE(Identity);

struct ResidualBlockImpl : torch::nn::Module {
  ResidualBlockImpl(int64_t in_features)
      : conv1(torch::nn::Conv2dOptions(in_features, in_features, 3)),
        norm1(in_features),
        conv2(torch::nn::Conv2dOptions(in_features, in_features, 3)),
        norm2(in_features) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("norm1", norm1);
    register_module("norm2", norm2);
  }

  torch::Tensor forward(torch::Tensor x) {
    auto y = x.clone();
    y = torch::reflection_pad2d(y, {1, 1, 1, 1});
    y = torch::relu(norm1(conv1(y)));
    y = torch::reflection_pad2d(y, {1, 1, 1, 1});
    y = norm2(conv2(y));
    return x + y;
  }

  torch::nn::Conv2d conv1, conv2;
  torch::nn::InstanceNorm2d norm1, norm2;
};
TORCH_MODULE(ResidualBlock);

struct EncoderImpl : torch::nn::Module {
  EncoderImpl(int64_t in_nc, int64_t ngf = 64)
      : conv1(torch::nn::Conv2dOptions(in_nc, ngf, 7)),
        norm1(ngf),
        conv2(torch::nn::Conv2dOptions(ngf, 2 * ngf, 3).stride(2).padding(1)),
        norm2(2 * ngf),
        conv3(
            torch::nn::Conv2dOptions(2 * ngf, 4 * ngf, 3).stride(2).padding(1)),
        norm3(4 * ngf) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("norm1", norm1);
    register_module("norm2", norm2);
    register_module("norm3", norm3);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::reflection_pad2d(x, {3, 3, 3, 3});
    x = torch::relu(norm1(conv1(x)));
    x = torch::relu(norm2(conv2(x)));
    x = torch::relu(norm3(conv3(x)));
    return x;
  }

  torch::nn::Conv2d conv1, conv2, conv3;
  torch::nn::InstanceNorm2d norm1, norm2, norm3;
};
TORCH_MODULE(Encoder);

struct TransformerImpl : torch::nn::Module {
  TransformerImpl(int64_t n_styles, int64_t ngf, bool auto_id = true) {
    for (int64_t i = 0; i < n_styles; ++i)
      mlist->push_back(ResidualBlock(ngf * 4));
    if (auto_id) mlist->push_back(Identity());
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor label) {
    for (int64_t i = 0; i < label.size(1); ++i)
      if (label[0][i].item<int64_t>() == 1)
        return mlist[i]->as<ResidualBlock>()->forward(x);
    return mlist[label.size(1)]->as<Identity>()->forward(x);
  }
  torch::nn::ModuleList mlist;
};
TORCH_MODULE(Transformer);

struct DecoderImpl : torch::nn::Module {
  DecoderImpl(int64_t out_nc, int64_t ngf, int64_t n_residual_blocks = 5)
      : _n_residual_blocks(n_residual_blocks),
        residual_block(4 * ngf),
        convt1(torch::nn::ConvTranspose2dOptions(4 * ngf, 2 * ngf, 3)
                   .stride(2)
                   .padding(1)
                   .output_padding(1)),
        norm1(2 * ngf),
        convt2(torch::nn::ConvTranspose2dOptions(2 * ngf, ngf, 3)
                   .stride(2)
                   .padding(1)
                   .output_padding(1)),
        norm2(ngf),
        conv1(torch::nn::Conv2dOptions(ngf, out_nc, 7).bias(false)) {
    register_module("convt1", convt1);
    register_module("convt2", convt2);
    register_module("conv1", conv1);
    register_module("norm1", norm1);
    register_module("norm2", norm2);
  }
  // ngf=64,2ngf=128,4ngf=256
  torch::Tensor forward(torch::Tensor x) {
    for (int64_t i = 0; i < _n_residual_blocks; ++i) x = residual_block(x);
    x = torch::relu(norm1(convt1(x)));
    x = torch::relu(norm2(convt2(x)));
    x = torch::reflection_pad2d(x, {3, 3, 3, 3});
    x = torch::tanh(conv1(x));
    return x;
  }
  int64_t _n_residual_blocks;
  torch::nn::Conv2d conv1;
  torch::nn::ConvTranspose2d convt1, convt2;
  torch::nn::InstanceNorm2d norm1, norm2;
  ResidualBlock residual_block;
};
TORCH_MODULE(Decoder);

struct GeneratorImpl : torch::nn::Module {
  GeneratorImpl(int64_t in_nc, int64_t out_nc, int64_t n_styles, int64_t ngf) {
    encoder = Encoder(in_nc, ngf);
    transformer = Transformer(n_styles, ngf);
    decoder = Decoder(out_nc, ngf);
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor label) {
    auto e = encoder(x);
    auto t = transformer(e, label);
    auto d = decoder(t);
    return d;
  }
  Encoder encoder = nullptr;
  Transformer transformer = nullptr;
  Decoder decoder = nullptr;
};
TORCH_MODULE(Generator);

struct DiscriminatorImpl : torch::nn::Module {
  DiscriminatorImpl(int64_t in_nc, int64_t n_styles, int64_t ndf = 64)
      : conv1(torch::nn::Conv2dOptions(in_nc, ndf, 4).stride(2).padding(2)),
        conv2(torch::nn::Conv2dOptions(ndf, 2 * ndf, 4).stride(2).padding(2)),
        norm2(2 * ndf),
        conv3(
            torch::nn::Conv2dOptions(2 * ndf, 4 * ndf, 4).stride(2).padding(2)),
        norm3(4 * ndf),
        conv4(
            torch::nn::Conv2dOptions(4 * ndf, 8 * ndf, 4).stride(1).padding(2)),
        norm4(8 * ndf),
        fldiscriminator(torch::nn::Conv2dOptions(8 * ndf, 1, 4).padding(2)),
        aux_clf(torch::nn::Conv2dOptions(8 * ndf, n_styles, 4).padding(2)),
        leaky_relu(
            torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("fldiscriminator", fldiscriminator);
    register_module("aux_clf", aux_clf);
    register_module("norm2", norm2);
    register_module("norm3", norm3);
    register_module("norm1", norm4);
    register_module("leaky_relu", leaky_relu);
  }

  auto forward(torch::Tensor x) {
    x = leaky_relu(conv1(x));
    x = leaky_relu(norm2(conv2(x)));
    x = leaky_relu(norm3(conv3(x)));
    auto base = leaky_relu(norm4(conv4(x)));
    auto discrim = fldiscriminator(base);
    auto clf = aux_clf(base);
    return std::make_pair(discrim, clf);
  }

  torch::nn::Conv2d conv1, conv2, conv3, conv4, fldiscriminator, aux_clf;
  torch::nn::InstanceNorm2d norm2, norm3, norm4;
  torch::nn::LeakyReLU leaky_relu;
};
TORCH_MODULE(Discriminator);

#endif  //_MODELS_H_