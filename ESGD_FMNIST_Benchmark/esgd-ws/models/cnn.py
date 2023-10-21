import torch
import torch.nn as nn
import torch.nn.functional as F



# model 4 (no_reg, for FMNIST dataset)
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 2, 1) # in, out, kernel, stride, dilation
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256 * 5 * 5, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        out = self.classifier(x)
        return out

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps")
m = CNN().to(DEVICE)



# example model from PyTorch
# class CNN(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5) # 3 in, 6 out, kernel = 5; default stride = 1;
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5) # 6 in channels, 16 out channels; 5x5 kernel, stride = 1
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# m = CNN().to(DEVICE)




# conv1 = nn.Conv2d(1, 64, 3, 2, 1)

# print(conv1.in_channels)
# print(conv1.out_channels)
# print(conv1.kernel_size)
# print(conv1.stride)
# print(conv1.dilation)



