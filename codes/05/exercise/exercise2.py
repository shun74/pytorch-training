from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, dataset_dir):
        dir_path_resolved = Path(dataset_dir).resolve()
        dir_list = list(dir_path_resolved.glob("*"))
        self.img_list = []
        for i in dir_list:
            img_path_list = list(i.glob("*.png"))
            # ディレクトリ毎にファイルのリストの結合
            self.img_list += img_path_list
        
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
    
    