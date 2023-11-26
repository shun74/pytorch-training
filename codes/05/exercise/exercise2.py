import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.paths = list(self.dataset_dir.glob("*"))
        self.img_list = []

        for i in self.paths:
            self.img_path_list = list(i.glob("*"))
            self.img_list += self.img_path_list
        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        
        return img
    

if __name__ == "__main__":
    my_dataset = MyDataset("./data")
    print("===== problem1.1 =====")
    print(len(my_dataset))
    print("===== problem1.2 =====") 
    print(my_dataset[0].size)
    
    