import os
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms.functional as TF

class TimedImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.n = len([d for d in os.listdir(root_dir) if "." not in d])

    def __len__(self):
        return self.n
                                            
    def __getitem__(self, idx):
        scene_dir = self.root_dir + f"/{idx}"
        times = [f[:-4] for f in os.listdir(scene_dir) if ".jpg" in f]

        scene_dict = {}
        for t in times:
             image = Image.open(scene_dir+f"/{t}.jpg")
             tensor = TF.to_tensor(image)
             scene_dict[t] = tensor
        return scene_dict
