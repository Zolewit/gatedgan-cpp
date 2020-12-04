#include <assert.h>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#include "data.h"
#include "models.h"
#include "utils.h"

struct Option {
  int64_t _epoch = 0;
  int64_t n_epochs = 200;
  int64_t decay_epoch = 100;
  int64_t batch_size = 1;
  std::string data_root = "photo2fourcollection";
  int64_t load_size = 143;
  int64_t fine_size = 128;
  int64_t ngf = 64;
  int64_t ndf = 64;
  int64_t in_nc = 3;
  int64_t out_nc = 3;
  double lr = 0.0002;
  double lambda_a = 10.0;
  int64_t auto_encoder_constrain = 10;
  int64_t n_styles = 4;
  bool cuda = false;
  double tv_strength = 1e-6;
  bool use_lsgan = true;
  int64_t kLogInterval = 1;
} options;

template <typename Optimizer = torch::optim::Adam,
          typename OptimizerOptions = torch::optim::AdamOptions>
inline auto decay(Optimizer &optimizer, int64_t epo, const Option &opt)
    -> void {
  assert(opt.n_epochs - opt.decay_epoch > 0);
  for (auto &group : optimizer.param_groups()) {
    if (group.has_options()) {
      auto &options = static_cast<OptimizerOptions &>(group.options());
      double rate = 1.0 - static_cast<double>(
                              std::max(static_cast<int64_t>(0),
                                       epo + opt._epoch - opt.decay_epoch)) /
                              (opt.n_epochs - opt.decay_epoch);
      options.lr(opt.lr * rate);
    }
  }
}

int main() {
  torch::Device device(torch::kCPU);
  if (options.cuda && torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  } else
    std::cout << "CUDA is not available! Training on CPU." << std::endl;

  auto data_set = ImageDataset(options.data_root)
                      .map(torch::data::transforms::Normalize<>(
                          {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                      .map(torch::data::transforms::Stack<>());
  auto data_size = data_set.size().value();
  const int64_t batches_per_epoch =
      std::ceil(data_size / static_cast<double>(options.batch_size));
  auto data_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(data_set), options.batch_size);

  auto generator =
      Generator(options.in_nc, options.out_nc, options.n_styles, options.ngf);
  auto discriminator =
      Discriminator(options.in_nc, options.n_styles, options.ndf);
  generator->to(device);
  discriminator->to(device);

  auto criterion_gan = torch::nn::MSELoss();
  // 暂时不实现
  // if (options.use_lsgan)
  //    criterion_gan = torch::nn::MSELoss();
  // else
  //    criterion_gan = torch::nn::BCELoss();

  auto criterion_acgan = torch::nn::CrossEntropyLoss();
  auto criterion_rec = torch::nn::L1Loss();
  auto criterion_tv = TVLoss(options.tv_strength);

  auto optimizer_g = torch::optim::Adam(
      generator->parameters(),
      torch::optim::AdamOptions(options.lr).betas(std::make_tuple(0.5, 0.999)));
  auto optimizer_d = torch::optim::Adam(
      discriminator->parameters(),
      torch::optim::AdamOptions(options.lr).betas(std::make_tuple(0.5, 0.999)));

  auto input_a = torch::zeros({options.batch_size, options.in_nc,
                               options.fine_size, options.fine_size});
  auto input_b = torch::zeros({options.batch_size, options.out_nc,
                               options.fine_size, options.fine_size});

  auto batch = data_loader->begin();

  auto img = batch->data.to(device);
  auto target = batch->target.to(device);
  img.transpose_(0, 1);  //将patchsize维度与增加的维度互换
  auto source = img[0];
  auto style = img[1];

  auto result = discriminator(style);
  auto d_a_size = result.first.sizes();
  auto d_ac_size = result.second.sizes();

  auto class_label_b = torch::zeros({d_ac_size[0], d_ac_size[2], d_ac_size[3]})
                           .to(torch::kInt64);

  // auto autoflag_ohe = torch::zeros(options.n_styles + 1);
  // autoflag_ohe[options.n_styles] = 1;
  //与原版不同，flag全为0，则视为触发autoencoder
  auto autoflag_ohe = torch::zeros({1, options.n_styles});

  auto fake_label = torch::zeros(d_a_size);
  auto real_label = torch::zeros(d_a_size).fill_(0.99);

  auto fake_buffer = ReplayBuffer();

  generator->apply(weights_init_normal);
  discriminator->apply(weights_init_normal);

  FILE *fpWrite = fopen("log.txt", "a");
  if (fpWrite == NULL) return 0;
  for (int64_t epoch = options._epoch; epoch < options.n_epochs; ++epoch) {
    int64_t batch_index = 0;
    for (torch::data::Example<> &batch : *data_loader) {
      auto img = batch.data.to(device);
      auto style_label = batch.target.to(device);
      img.transpose_(0, 1);
      auto real_content = img[0];

      auto real_style = img[1];
      auto style_ohe =
          torch::nn::functional::one_hot(style_label, options.n_styles)
              .to(torch::kInt64);
      auto class_label =
          label2tensor(style_label, class_label_b).to(torch::kInt64);

      optimizer_d.zero_grad();
      auto gen_fake = generator(real_content, style_ohe);
      auto fake = gen_fake;
      // auto fake = fake_buffer.push_and_pop(gen_fake);

      auto out = discriminator(fake);
      auto out_gan = out.first;
      auto out_class = out.second;
      auto err_d_fake = criterion_gan(out_gan, fake_label);

      err_d_fake.backward();

      optimizer_d.step();

      optimizer_d.zero_grad();
      out = discriminator(real_style);
      out_gan = out.first;
      out_class = out.second;
      auto err_d_real_class =
          criterion_acgan(out_class, class_label) * options.lambda_a;
      auto err_d_real = criterion_gan(out_gan, real_label);
      auto err_d_real_total = err_d_real + err_d_real_class;
      err_d_real_total.backward();
      optimizer_d.step();

      auto err_d = (err_d_real + err_d_fake) / 2.0;

      optimizer_g.zero_grad();
      out = discriminator(gen_fake.detach());
      out_gan = out.first;
      out_class = out.second;

      auto err_gan = criterion_gan(out_gan, real_label);
      auto err_class =
          criterion_acgan(out_class, class_label) * options.lambda_a;
      auto err_tv = criterion_tv(gen_fake);

      auto err_g_tot = err_gan + err_class + err_tv;
      err_g_tot.backward();
      optimizer_g.step();

      optimizer_g.zero_grad();
      auto identity = generator(real_content, autoflag_ohe);
      auto err_ae = criterion_rec(identity, real_content) *
                    options.auto_encoder_constrain;
      err_ae.backward();
      optimizer_g.step();

      if (batch_index++ % options.kLogInterval == 0) {
        std::printf("[%I64d/%I64d][%I64d/%I64d] D_loss: %.4f | G_loss: %.4f\n",
                    epoch, options.n_epochs, batch_index, batches_per_epoch,
                    err_d.item<float>(), err_g_tot.item<float>());
        std::fprintf(fpWrite,
                     "[%I64d/%I64d][%I64d/%I64d] D_loss: %.4f | G_loss: %.4f\n",
                     epoch, options.n_epochs, batch_index, batches_per_epoch,
                     err_d.item<float>(), err_g_tot.item<float>());
      }
    }
    decay(optimizer_g, epoch, options);
    decay(optimizer_d, epoch, options);

    torch::save(generator, "netG.pth");
    torch::save(discriminator, "netD.pth");
  }
  fclose(fpWrite);

  return 0;
}