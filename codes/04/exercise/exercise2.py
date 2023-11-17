import torch
from torch import nn


if __name__=="__main__":

    # problem 1
    _in = torch.ones((32, 1024))
    print("===== problem 1 =====")
    print(repr(_in.size()))

    # problem 2
    fc = nn.Linear(in_features=1024, out_features=256)
    out = fc(_in)
    print("===== problem 2 =====")
    print(repr(out.size()))

    # problem 3
    fc2 = nn.Linear(in_features=1024, out_features=2048)
    print("===== problem 3 =====")
    print(repr(fc2(_in).size()))

    # appendix
    print("===== appendix =====")
    print(repr(out.view(-1, 16, 16).size()))
    # just in case
    # print(repr(out.contiguous().view(-1, 16, 16).size()))
