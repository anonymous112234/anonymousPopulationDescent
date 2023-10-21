import torch
import torch.nn as nn
import torch.nn.functional as F


# # model 4 (no_reg, for FMNIST dataset)
# class CNN(nn.Module):


# model #6 CIFAR better 
class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            # nn.BatchNorm2d(),

            # nn.Dropout(0.2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            # nn.BatchNorm2d(),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            # nn.ReLU(),
            # nn.BatchNorm2d(),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            # nn.BatchNorm2d(),

            nn.MaxPool2d(4, 4),
            # nn.Dropout(0.2)
            )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64, 256),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(256, 10),
            )

    def forward(self, x):

        x = self.layers(x)
        x = self.classifier(x)
        # out = nn.functional.softmax(x)
        out = x

        return out

        # return x


DEVICE = torch.device("mps")
m = CNN().to(DEVICE)
