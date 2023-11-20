import torch


if __name__=="__main__":

    # 入力のテンソルを定義
    _in = torch.ones((32, 8, 16, 16))

    # 形状を変更
    # (B, C, W, H) -> (B, N)
    out = _in.view(32, 8*16*16)

    # 確認
    print("===== before =====")
    print(repr(_in.size()))
    print("===== after =====")
    print(repr(out.size()))
