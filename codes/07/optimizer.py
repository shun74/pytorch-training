import torch
import torch.nn as nn
import torch.optim as optim

# モデルを定義
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

if __name__=="__main__":
    torch.manual_seed(0)
    x = torch.randn(100, 1)
    y = 3 * x + 0.5 * torch.randn(100, 1)

    model = MyModel()

    # 損失関数（平均二乗誤差）
    criterion = nn.MSELoss()

    # 最適化関数 (SGD)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 学習
    epochs = 100
    losses = []
    for epoch in range(epochs):
        model.train()

        # パラメータの勾配を初期化
        optimizer.zero_grad() 

        # 順伝搬・逆伝搬・パラメータ更新
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch + 1: <3}, Loss: {loss.item():.4f}')