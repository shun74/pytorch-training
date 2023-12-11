from torch.utils.data import DataLoader
from dataset import datasets

# データローダーからデータを受け取る
train_data, test_data = datasets()
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

if __name__=="__main__":
    # イテレータを作成
    train_iter = iter(train_loader)
    
    # 次のバッチを取得
    image, labels = next(train_iter)
    
    print("Image shape: ", image.shape) # [batch_size, channels, height, width]
    print("Labels: ", labels)