import os
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as transforms


class TimedImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.n = len([d for d in os.listdir(root_dir) if "." not in d])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if isinstance(idx, slice) or isinstance(idx, range):
            ifnone = lambda a, b: b if a is None else a
            scene_dicts = []
            for x in range(ifnone(idx.start, 0),
                           ifnone(idx.stop, 0),
                           ifnone(idx.step, 1)):
                scene_dicts.append(self[x])
            ret_dict = {}
            for key in scene_dicts[0].keys():
                ret_dict[key] = torch.stack([x[key] for x in scene_dicts])
            return ret_dict

        scene_dir = self.root_dir + f"/{idx}"
        times = [f[:-4] for f in os.listdir(scene_dir) if ".jpg" in f]

        scene_dict = {}
        for t in times:
            hour, minute = t.split('_')
            image = Image.open(scene_dir + f"/{t}.jpg")
            scene_dict[int(hour)] = self.transform(image)
        return scene_dict


def prefix_sum(lengths):
    result = []
    for i, x in enumerate(lengths):
        if i == 0:
            result.append(x)
        else:
            result.append(x + result[i - 1])
    return result


if __name__ == '__main__':
    dataset = TimedImageDataset(r'E:\TimeLapseVDataDownsampled')
