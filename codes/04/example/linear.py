import torch
from torch import nn


if __name__=="__main__":

    # 入力のテンソルを定義
    _in = torch.ones((32, 1280))

    # linear 定義 ＆ 適用
    fc = nn.Linear(in_features=1280, out_features=256, bias=True)
    print(repr(fc(_in).size()))
