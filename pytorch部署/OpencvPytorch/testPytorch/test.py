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
dummy_input = torch.randn(1, 3, 32, 32).to(device)
traced_cell = torch.jit.trace(model, dummy_input)
img = cv2.imread('./dog.jpg')
img = cv2.resize(img, (32, 32))
print(type(img))
new_img = torch.tensor(img/255, dtype=torch.float)
new_img = new_img.permute(2, 0, 1)
new_img = new_img.unsqueeze(0)
print(traced_cell(new_img.to(device)))
traced_cell.save("tests.pth")