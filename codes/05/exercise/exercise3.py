import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch 
from torchvision import transforms
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.paths = list(self.dataset_dir.glob("*"))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.img_list = []

        for i in self.paths:
            self.img_path_list = list(i.glob("*"))
            self.img_list += self.img_path_list
        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        print(img_path)
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        img_path = Path(img_path)
        parts = img_path.parts
        label = parts[-2]

        return img_tensor, label
    

if __name__ == "__main__":
    my_dataset = MyDataset("./data")
    img, label = my_dataset[0]
    print("===== problem1.1 =====")
    print(img.size())
    print("===== problem1.2 =====")
    print(label)