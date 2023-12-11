import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import datasets
from model import CNN

# データローダーからデータを受け取る (参考：dataloader.py)
train_data, test_data = datasets()
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# モデルの定義
model = CNN()

# 損失関数
criterion = nn.CrossEntropyLoss()

# 最適化関数
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(1):  
    train_loss = 0
    val_loss = 0

    model.train()
    for images, labels in train_loader:
        # 勾配を初期化
        optimizer.zero_grad()

        train_outputs = model(images) # [batch_size, label]
        loss = criterion(train_outputs, labels)
        train_loss += loss.item() 

        # 誤差逆伝播
        loss.backward()

        # 重みを更新
        optimizer.step()
        
    avg_train_loss = train_loss / len(train_loader)
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            test_outputs = model(images) # [batch_size, label]
            loss = criterion(test_outputs, labels)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)

if __name__=="__main__":
    print("Train Loss: ", avg_train_loss)
    print("Validation Loss: ", avg_val_loss)