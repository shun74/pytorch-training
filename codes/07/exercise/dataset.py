import torchvision
import torchvision.transforms as transforms

def datasets():
    # データセットの読み込み
    train_data = torchvision.datasets.CIFAR10(root="./", 
                                            train=True,
                                            transform=transforms.ToTensor(), 
                                            download=True)

    test_data = torchvision.datasets.CIFAR10(root="./", 
                                            train=False, 
                                            transform=transforms.ToTensor(), 
                                            download=True)
    
    return train_data, test_data

if __name__=="__main__":
    train_data, test_data = datasets()
    image, label = train_data[0]
    print("image size: ", image.size())
    print("image label: ", label)
