import os
from torch.utils.data import Dataset, DataLoader
import torch
from utils import load_image_as_vec


class GlazeDataset (Dataset):
    def __init__(self, material_dict, raw_data, recipes, image_dir='images'):
        self.recipes = recipes
        self.image_dir = image_dir
        self.material_dict = material_dict
        self.raw_data = raw_data

    def get_recipe(self, idx):
        return torch.as_tensor(self.recipes.as_matrix[idx]).float()

    def get_composition(self, idx)
        d = self.raw_data[idx]
        return 

    def get_image(self, idx):
        d = self.raw_data[idx]
        image_name = os.path.join(
            self.image_dir, d['selectedImage']['filename'])
        return load_image_as_vec(image_name)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.get_image(idx), self.get_recipe(idx)
