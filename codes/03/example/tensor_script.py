import torch
import numpy as np

# 0 埋めの Tensor 作成
# 形状指定はリスト、タプル、数値、のどれでも問題なし
zero_tensor1 = torch.zeros((2, 4))
zero_tensor2 = torch.zeros([2, 4])
zero_tensor3 = torch.zeros(2, 4)
print(zero_tensor1, "\n", zero_tensor2, "\n", zero_tensor3)

# 任意の値で埋めた Tensor 作成
# 形状指定はリスト、タプルのみ、＋埋めたい値
any_tensor1 = torch.full([2, 4], 3.14)
any_tensor2 = torch.full((2, 4), 3.14)
print(any_tensor1, "\n", any_tensor2)

# numpy からの Tensor に変換
arr = np.full((2, 4), 3.14)
shared_tensor = torch.from_numpy(arr)
private_tensor = torch.tensor(arr)
print("before")
print(shared_tensor, "\n", private_tensor)
arr[0] = -1
print("after")
print(shared_tensor, "\n", private_tensor)

# 形状確認 & 軸入れ替え
# permute( 対象のテンソル , 対象テンソルの並び替え後の軸(0-index) )
zero_tensor4 = torch.zeros((2, 4, 8))
print(zero_tensor4.size(), torch.permute(zero_tensor4, (2, 0, 1)).size())
