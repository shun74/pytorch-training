import torch
from torch.utils.data import DataLoader

from dataset import datasets
from model import CNN

# データローダーからデータを受け取る (参考：dataloader.py)
train_data, test_data = datasets()
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# モデルの定義
model = CNN()

# 各バッチにおけるモデルの出力サイズのリスト
train_outputs_list = []
test_outputs_list = []

for epoch in range(1):      
    # 訓練データでモデルの出力
    model.train()
    for images, labels in train_loader:
        train_outputs = model(images) # [batch_size, label]
        train_outputs_list.append(train_outputs)
    
    # テストデータでモデルの出力
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            test_outputs = model(images) # [batch_size, label]
            test_outputs_list.append(test_outputs)

if __name__=="__main__":
    # 1番目のバッチにおけるモデルの出力サイズ
    print("Train Output Size:", train_outputs_list[0].shape)
    print("Test Output Size:", test_outputs_list[0].shape)