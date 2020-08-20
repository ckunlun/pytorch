#include <torch/script.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// 有人说调用的顺序有关系，我这好像没啥用~~

int main()
{
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Predicting on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else {
        std::cout << "Predicting on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    //Init model
    std::string model_pb = "tests.pth";
    auto module = torch::jit::load(model_pb);
    module.to(at::kCUDA);

    auto image = cv::imread("dog.jpg", cv::ImreadModes::IMREAD_COLOR);
    cv::Mat image_transfomed;
    cv::resize(image, image_transfomed, cv::Size(32, 32));

    // convert to tensort
    torch::Tensor tensor_image = torch::from_blob(image_transfomed.data,
        { image_transfomed.rows, image_transfomed.cols,3 }, torch::kByte);
    tensor_image = tensor_image.permute({ 2,0,1 });
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);
    tensor_image = tensor_image.unsqueeze(0);
    tensor_image = tensor_image.to(at::kCUDA);
    torch::Tensor output = module.forward({ tensor_image }).toTensor();
    auto max_result = output.max(1, true);
    auto max_index = std::get<1>(max_result).item<float>();
    std::cout << output << std::endl;
    //return max_index;
    return 0;

}
