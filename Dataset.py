import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class HazeDataset(Dataset):
        def __init__(self, root_dir, transforms):
            self.transforms = transforms
            self.root_dir = root_dir
            self.image_list = self.get_image_pairs()
            print("Total data examples:", len(self.image_list))

        def get_image_pairs(self):
            image_list = []
            hazy_root = os.path.join(self.root_dir, 'hazy')
            ground_truth_root = os.path.join(self.root_dir, 'GT')
            
            hazy_folders = os.listdir(hazy_root)
            for folder in hazy_folders:
                hazy_folder_path = os.path.join(hazy_root, folder)
                ground_truth_folder_path = os.path.join(ground_truth_root, folder)
                
                hazy_images = os.listdir(hazy_folder_path)
                for hazy_img_name in hazy_images:
                    hazy_img_path = os.path.join(hazy_folder_path, hazy_img_name)
                    ground_truth_img_path = os.path.join(ground_truth_folder_path, hazy_img_name)
                    image_list.append((ground_truth_img_path, hazy_img_path))
            
            random.shuffle(image_list)
            return image_list

        def __len__(self):
            return len(self.image_list)

        def __getitem__(self, idx):
            ground_truth_path, hazy_img_path = self.image_list[idx]
            ground_truth_img = self.transforms(Image.open(ground_truth_path))
            hazy_img = self.transforms(Image.open(hazy_img_path))
            return ground_truth_img, hazy_img