import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch 
from torchvision import transforms
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.img_list = list(self.dataset_dir.glob("*"))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        img_tensor = self.transform(img)

        img_path = Path(img_path)
        parts = img_path.parts
        label = parts[-1][:3]

        return img_tensor, label
    

if __name__ == "__main__":
    my_dataset = MyDataset("./data")
    print("---Number of files in the dataset---")
    print(len(my_dataset))
    img, label = my_dataset[0]
    print("img.size :",img.size())
    print("label    :",label)