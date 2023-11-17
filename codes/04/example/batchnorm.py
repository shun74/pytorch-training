import torch
from torch import nn


if __name__=="__main__":
    
    # 入力用のテンソル定義
    _in = torch.ones((32, 256, 64, 64))

    # batchnorm 定義 ＆ 適用
    norm = nn.BatchNorm2d(num_features=256)
    print(repr(norm(_in).size()))

    # 確認用
    _in2 = torch.tensor([
        [3., 2., 5.],
        [16., 43., 1.],
        [18., 3.1, 56.]
    ]).unsqueeze(dim=0).unsqueeze(dim=0)
    norm2 = nn.BatchNorm2d(num_features=1, affine=False)
    print("===== before =====")
    print(repr(_in2))
    print("===== after =====")
    print(repr(norm2(_in2)))
