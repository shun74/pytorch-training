from torchvision import transforms, datasets

def cifar_datasets():
    # データセットの読み込み
    train_data = datasets.CIFAR10(root="../", 
                                            train=True,
                                            transform=transforms.ToTensor(), 
                                            download=True)

    test_data = datasets.CIFAR10(root="../", 
                                            train=False, 
                                            transform=transforms.ToTensor(), 
                                            download=True)
    
    return train_data, test_data

if __name__=="__main__":
    train_data, test_data = cifar_datasets()

    # 訓練データの1枚目を取得
    image, label = train_data[0]
    print("image size: ", image.size())
    print("image label: ", label)