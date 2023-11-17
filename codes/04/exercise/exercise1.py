import torch
from torch import nn


if __name__=="__main__":
    _in = torch.ones((32, 3, 128, 128))

    conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
    print("===== problem1 =====")
    print(repr(conv1(_in).size()))

    conv2 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=2, padding=1)
    print("===== problem2 =====")
    print(repr(conv2(_in).size()))

    conv3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1)
    print("===== problem1 =====")
    print(repr(conv3(_in).size()))

    conv4 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=2, padding=2)
    print("===== problem2 =====")
    print(repr(conv4(_in).size()))
