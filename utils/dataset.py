import os

import cv2
import torch
import torch.utils.data as data
from pathlib import Path
import torchvision.transforms as transform
import torchvision.transforms.functional as F

class MyDataset(data.Dataset):
    def __init__(self, cfg, is_train):
        super().__init__()
        self.className2idx = {
            'banana': 0, 
            'bareland': 1,
            'inundated': 1,
            'carrot': 2,
            'corn': 3,
            'dragonfruit': 4,
            'garlic': 5,
            'guava': 6,
            'peanut': 7,
            'pineapple': 8,
            'pumpkin': 9,
            'rice': 10,
            'soybean': 11,
            'sugarcane': 12,
            'tomato': 13
        }

        compose = [transform.ToTensor()]
        if is_train:
            if cfg['train']['image_size'] != -1:
                compose.append(transform.RandomResizedCrop(cfg['train']['image_size'], scale=(0.05, 1.0)))
            compose.append(transform.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
            compose.append(transform.RandomRotation(15))
            compose.append(transform.RandomHorizontalFlip())
        else:
            if cfg['val']['image_size'] != -1:
                compose.append(transform.Resize((cfg['val']['image_size'], cfg['val']['image_size'])))
        compose.append(transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transform.Compose(compose)  

        data_folder = Path(os.path.join(cfg['data']['dataset_path'], cfg['data']['sub_folder_train'] if is_train else cfg['data']['sub_folder_val']))
        self.data_list = list(data_folder.rglob(cfg['data']['data_format']))
        self.len = len(self.data_list)

    def __getitem__(self, index):
        data_path = str(self.data_list[index])
        class_name_end_idx = data_path.rfind('/')
        class_name_start_idx = data_path.rfind('/', 0, class_name_end_idx)+1
        class_name = data_path[class_name_start_idx:class_name_end_idx]

        img = cv2.imread(data_path)
        img = self.transform(img)
        label = torch.tensor(self.className2idx[class_name])
        data = {
            'img': img,
            'label': label
        }
        return data

    def __len__(self):
        return self.len