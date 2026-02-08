import numpy as np
import os, glob, cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PathPlanningDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_list = []
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        search_path = os.path.join(root_dir, "*", "*", "*", "coords.txt")
        for file_path in glob.glob(search_path):
            pair_folder = os.path.dirname(file_path)
            num_folder = os.path.dirname(pair_folder)

            map_path = os.path.join(num_folder, "map.png")
            points_path = os.path.join(pair_folder, "points.png")
            promising_path = os.path.join(pair_folder, "promising.png")

            img = cv2.imread(promising_path, cv2.IMREAD_GRAYSCALE)
            if np.mean(img) == 255:
                continue

            if (
                os.path.exists(map_path) 
                and os.path.exists(points_path) 
                and os.path.exists(promising_path)
            ):
                self.data_list.append({
                    "map": map_path,
                    "points": points_path,
                    "target": promising_path,
                })
    
        if len(self.data_list) > 0:
            print(f"Found {len(self.data_list)} valid data samples!")
        else:
            print(f"No valid data samples found in {root_dir}!")

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]

        map_img = Image.open(item["map"]).convert("L")
        points_img = Image.open(item["points"]).convert("RGB")
        target_img = Image.open(item["target"]).convert("L")

        map_tensor = self.transform(map_img)
        points_tensor = self.transform(points_img)
        input_tensor = points_tensor * map_tensor
        target_tensor = self.transform(target_img)
        target_tensor = (target_tensor < 0.95).float()

        return input_tensor, target_tensor
    
if __name__ == "__main__":
    dataset = PathPlanningDataset(root_dir="datasets")