import os
from skimage import io, transform
from skimage.color import rgba2rgb
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms

transform_norm = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class GlazeDataset (Dataset):
    def __init__(self, material_dict, raw_data, recipes, image_dir='images'):
        self.recipes = recipes
        self.image_dir = image_dir
        self.material_dict = material_dict
        self.raw_data = raw_data

    def get_recipe(self, idx):
        return torch.as_tensor(self.recipes.as_matrix[idx])

    def get_image(self, idx):
        d = self.raw_data[idx]
        image_name = os.path.join(
            self.image_dir, d['selectedImage']['filename'])
        image = io.imread(image_name)
        image = transform.resize(
            image, (32, 32), mode='constant', anti_aliasing=True)
        if image.shape[2] == 4:
            image = rgba2rgb(image)
        image = transform_norm(image)
        return image

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.get_image(idx).float(), self.get_recipe(idx).float()
