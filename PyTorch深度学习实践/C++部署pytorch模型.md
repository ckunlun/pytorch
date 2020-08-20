[toc]

# C++部署pytorch模型

## 前言

项目需要将pytorch训练好的网络用c++调用，在正式开始项目之前，在网上查了各种资料，共有三种实现方法：
* 直接将网络从最基础的CNN模块用C++实现；
* 将网咯模型和参数保存，然后使用opencv的DNN模块加载，这个方法tensorflow、torch等其他网络架构也能用，具体包含哪些下文会给出；
* 使用pytorch官网提供的c++接口：LibTorch。其原理也是保存网络模型和参数，然后用LibTorch进行加载。
由于第一项c++从第层撸起太过硬核，自己水平有限，此处就不做介绍。大佬们可以自己尝试。此处只介绍opencv和LibTorch实现的方法。
```bash
运行环境：
win10 64位
cuda 10.2
pytorch 1.6.0
torchvision 0.7
opencv 4.3
vs2019
LibTorch 1.6
ps:
pytorch相关软件都是直接在官网下载的最新版本。
```
## 训练一个简单的pytorch网络
首先，参考pytorch官方文档中[训练一个分类器](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)的代码，训练一个简单的图像分类器。代码如下：

```python

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.onnx
import torchvision
import torchvision.transforms as transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 32, 3)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(outputs)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

```
上述代码相对于官方文档的代码，仅仅是增加了卷积层和利用GPU进行训练，且输出结果未经处理，只是简单输出各个类别的概率值。
训练完网络之后，将网络保存，代码如下：
```Python
# 保存网络结构和参数

# 方法1：保存网络结构和参数
PATH = './cifar_net.pth'
torch.save(net, PATH)

# 方法2：保存网络参数
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# 方法3：导出网络到ONNX
dummy_input = torch.randn(1, 3, 32, 32).to(device)
torch.onnx.export(net, dummy_input, "torch.onnx")

# 方法4：保存网络位TORCHSCRIPT
dummy_input = torch.randn(1, 3, 32, 32).to(device)
traced_cell = torch.jit.trace(net, dummy_input)
traced_cell.save("tests.pth")
```
上述四种保存方法本文主要使用方法3和方法4，具体应用方式在下文会详细说明。此处简单说一下一个比较坑的地方：我开始以为使用方法1保存的网络可以像tensorflow那样直接用load函数导入，自动重建出原始网络架构，但是试验后才发现，要想成功导入，还需要将定义网络的类也放在对应的py文件中，这有点。。。
方法1导入示例代码如下：

```python
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
PATH = './cifar_net.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 32, 3)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = torch.load(PATH)
```
***到此，所有的准备工作已经完成。***

## 利用opencv加载训练好的模型和网络

参考链接：
[OpenCV4.0 运行快速风格迁移（Torch）](https://blog.csdn.net/juebai123/article/details/86545556)
[opencv官方文档](https://docs.opencv.org/4.3.0/d6/d0f/group__dnn.html#ga7faea56041d10c71dbbd6746ca854197)
[OpenCV加载Pytorch模型出现Unsupported Lua type 解决方法](https://www.cnblogs.com/DragonStart/p/12851987.html)
[将模型从PYTORCH导出到ONNX并使用ONNX RUNTIME运行（官网链接）](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
这个方法算是一个比较常用的方法，而且可以用到许多深度学习框架上面：
根据opencv官方文档中的说明，可以支持以下框架：Caffe，Darknet，Onnx，Tensorflow，Torch等。但是很可惜，没有我用的pytoch，但是根据第三个参考链接中的方法，可以利用ONNX实现曲线救国。首先利用保存模型方法3所示的办法，将网络和参数保存为对应的格式。然后使用opencv提供的`Net cv::dnn::readNetFromONNX	(	const String & 	onnxFile	)	`函数读取保存好的网络。代码实现如下：

```cpp
//测试opencv加载pytorch模型
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;
#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;


int main()
{
	String modelFile = "./torch.onnx";
	String imageFile = "./dog.jpg";

	dnn::Net net = cv::dnn::readNetFromONNX(modelFile); //读取网络和参数
	
	Mat image = imread(imageFile); // 读取测试图片
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	Mat inputBolb = blobFromImage(image, 0.00390625f, Size(32, 32), Scalar(), false, false); //将图像转化为正确输入格式

	net.setInput(inputBolb); //输入图像

	Mat result = net.forward(); //前向计算

	cout << result << endl;
}


```
上述代码就是对第一个参考链接的代码进行了简化，且将输入网络的模型从torch改成ONNX格式。
运行结果如下：

```bash
[-0.19793352, -4.0697966, 1.2769811, 2.7011304, 0.22390884, 1.9039617, -0.47333384, -0.15912014, 0.32441139, -2.4327304]

```
如果需要部署其他深度学习框架的网络，执行步骤基本类似。

## 利用pytorch官方提供的LibTorch加载训练好的模型和网络
参考链接：
[windows+VS2019+PyTorchLib配置使用攻略](https://www.jianshu.com/p/2371ee8b45f0)
[C++调用pytorch，LibTorch在win10下的vs配置和cmake的配置](https://zhuanlan.zhihu.com/p/68901339)
[在C ++中加载TORCHSCRIPT模型官网链接](https://pytorch.org/tutorials/advanced/cpp_export.html)
此处首先说明一下将pytroch保存为TORCHSCRIPT的方法有两种，一种是追踪式，另一种是脚本式。具体介绍见[官方文档](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)，理论上此方法两种保存方式都行，方法4中的是追踪式的方法，此文使用此方法。
首先按照第一个参考链接中的方法配置LibTorch环境，然后复制粘贴其中的示例代码，进行测试，但是我个人在运行的时候`ToTensor(image).to(at::kCUDA);`这个语句报错了，提示未定义ToTensor()，这句话的功能也很简单，就是将普通图像格式转化为模型输入需要的格式，于是我又根据第二个参考链接将转化代码进行了修改，代码如下：

```cpp
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

```
运行结果如下：

```bash
CUDA available! Predicting on GPU.
 1.0824 -4.6106  1.0189  2.9937  1.4570  1.4964 -1.3164 -0.7753  0.4567 -3.2543
[ CUDAFloatType{1,10} ]
```