#ifndef _DATA_H_
#define _DATA_H_

#include <torch/torch.h>

#include <opencv2/opencv.hpp>

const int64_t kImageSize = 128;

class ImageDataset : public torch::data::datasets::Dataset<ImageDataset> {
  std::vector<std::string> vec_img;
  std::vector<std::pair<int64_t, std::string>> vec_label_style;

  // get label in url
  std::string get_label(const std::string& filename) {
    int64_t found = filename.find_last_of("\\");
    std::string filename_path = filename.substr(0, found);
    found = filename_path.find_last_of("\\");
    std::string label = filename_path.substr(found + 1);
    return label;
  }

  // get (label_num,style_url)
  std::vector<std::pair<int64_t, std::string>> label(
      const std::vector<std::string>& vec_style) {
    std::vector<std::pair<int64_t, std::string>> vec_label_style;
    int64_t index = 0;
    std::unordered_map<std::string, int64_t> map_label;
    for (auto&& style : vec_style) {
      std::string label = get_label(style);
      if (0 == map_label.count(label)) map_label[label] = index++;
      vec_label_style.emplace_back(std::make_pair(map_label[label], style));
    }
    return vec_label_style;
  }

  //随机剪裁
  void random_crop(cv::Mat& mat, int64_t size) {
    assert(mat.cols >= size && mat.rows >= size && size > 0);
    int64_t point_x =
        torch::randint(0, mat.cols - size + 1, {1}).item<int64_t>();
    int64_t point_y =
        torch::randint(0, mat.rows - size + 1, {1}).item<int64_t>();
    cv::Rect myROI(point_x, point_y, size, size);
    mat(myROI).copyTo(mat);
  }

  //随机水平翻转
  void random_horizontal_flip(cv::Mat& mat) {
    if (torch::randint(0, 10, {1}).item<int64_t>() < 5) cv::flip(mat, mat, -1);
  }

  /**1.Resize
   * 2.RandomCrop
   * 3.RandomHorizontalFlip
   * 4.ToTensor
   **/
  torch::Tensor transform(cv::Mat mat) {
    cv::resize(mat, mat, cv::Size(143, 143), cv::INTER_CUBIC);
    random_crop(mat, kImageSize);
    random_horizontal_flip(mat);
    std::vector<cv::Mat> channels(3);
    cv::split(mat, channels);

    // BGR2RGB,Mat2Tensor
    auto R = torch::from_blob(channels[2].ptr(), {kImageSize, kImageSize},
                              torch::kUInt8);
    auto G = torch::from_blob(channels[1].ptr(), {kImageSize, kImageSize},
                              torch::kUInt8);
    auto B = torch::from_blob(channels[0].ptr(), {kImageSize, kImageSize},
                              torch::kUInt8);

    auto tdata = torch::cat({R, G, B})
                     .view({3, kImageSize, kImageSize})
                     .to(torch::kFloat);

    //除以255
    for (int64_t i = 0; i < 3; ++i) tdata[i].div_(255);

    return tdata;
  }

 public:
  ImageDataset(const std::string root, const std::string model = "train") {
    // get img list
    std::string pattern_img = root + "\\" + model + "Content" + "\\" + "*";
    cv::glob(pattern_img, vec_img);
    sort(vec_img.begin(), vec_img.end());

    // get list of (label_num,style_url)
    std::string pattern_style = root + "\\" + model + "Styles" + "\\" + "*";
    std::vector<std::string> vec_style;
    cv::glob(pattern_style, vec_style, true);
    vec_label_style = label(vec_style);
    sort(vec_label_style.begin(), vec_label_style.end());
  }

  torch::data::Example<> get(size_t index) {
    torch::Tensor img =
        transform(cv::imread(vec_img[index % vec_img.size()]));  // 3*a*a
    std::pair<int64_t, std::string> selection =
        vec_label_style[torch::randint(0, vec_label_style.size(), {1})
                            .item<int64_t>()];
    torch::Tensor style = transform(cv::imread(selection.second));
    // 2 tensors->1
    img = torch::unsqueeze(img, 0);
    style = torch::unsqueeze(style, 0);
    torch::Tensor data = torch::cat({img, style});
    torch::Tensor label = torch::tensor(selection.first);
    return {data, label};
  }

  torch::optional<size_t> size() const {
    return std::max(vec_img.size(), vec_label_style.size());
  }
};

class ReplayBuffer {
  std::vector<torch::Tensor> vec_data;
  int64_t _max_size;

 public:
  ReplayBuffer(int64_t max_size = 3) : _max_size(max_size) {
    // vec_data.reserve(_max_size);
  }

  torch::Tensor push_and_pop(torch::Tensor data) {
    torch::Tensor to_return = torch::ones_like(data);
    for (int64_t i = 0; i < data.size(0); ++i) {
      auto element = data[i].clone();
      if (static_cast<int64_t>(vec_data.size()) < _max_size) {
        vec_data.push_back(element);
        to_return[i] = element.clone();
      } else {
        if (torch::randint(0, 10, {1}).item<int64_t>() < 5) {
          int64_t index = torch::randint(0, _max_size, {1}).item<int64_t>();
          to_return[i] = vec_data[index].clone();
          vec_data[index] = element.clone();
        } else
          to_return[i] = element.clone();
      }
    }
    return to_return;
  }
};

#endif  // _DATA_H_
