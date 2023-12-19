# モジュールのインポート
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# dataset.py内のdatasets関数をインポート
from dataset import cifar_dataset
# model.py内のCNNクラスをインポート
from model import CNN

# 保存されたモデルのパス
model_path = 'cifar_cnn.pth'

# データローダーからデータを受け取る
_, test_data = cifar_dataset()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# モデルの定義
model = CNN()

# 保存されたモデルの読み込み
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

if __name__=="__main__":
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss = val_loss / len(test_loader)
    avg_val_acc = val_acc / len(test_loader.dataset)
    
    print('val_loss: {:.4f}, val_acc: {:.4f}'.format(avg_val_loss, avg_val_acc))