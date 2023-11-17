from torch import nn
from torch.nn import Module


class ExerciseModel(Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, \
                              stride=8, padding=2)
        self.norm = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=256*16*16, out_features=64) # 16 = 128 / 8

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = x.view(-1, 256*16*16)
        x = self.fc(x)

        return x
