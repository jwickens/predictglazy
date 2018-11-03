from __future__ import print_function, division
import os
import json
from skimage import io, transform
from skimage.color import rgba2rgb
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class GlazyDataset (Dataset):
  def __init__ (self, json_file='data.json', image_dir='images', transform=None):
    self.image_dir = image_dir
    self.transform = transform
    with open(json_file) as f:
      self.raw_data = json.load(f)
    self.process_materials()

  def process_materials (self):
    self.materials = {}
    self.materials_by_norm = {} 
    i = 0
    for data in self.raw_data:
      if 'materialComponents' in data:
        for material in data['materialComponents']:
          mid = material['material']['id']
          name = material['material']['name']
          if mid in self.materials:
            if self.materials[mid]['name'] != name:
              print('id mismatch for material %s' % name)
          else:
            self.materials[mid] = { 'name': name, 'norm_id': i }
            self.materials_by_norm[i] = mid
            i += 1
      else:
        print('no material components in %s' % data['id'])
    self.out_dim = len(self.materials.keys())
    print('total materials %i' % self.out_dim)

  def get_recipe (self, idx):
    d = self.raw_data[idx]
    recipe = torch.zeros(self.out_dim)
    if 'materialComponents' in d:
      for material in d['materialComponents']:
        mid = material['material']['id']
        nid = self.materials[mid]['norm_id']
        recipe[nid] = float(material['percentageAmount'])
    return recipe / 100

  def get_image (self, idx):
    d = self.raw_data[idx]
    image_name = os.path.join(self.image_dir, d['selectedImage']['filename'])
    image = io.imread(image_name)
    image = transform.resize(image, (32, 32), mode='constant', anti_aliasing=True)
    if image.shape[2] == 4:
      image = rgba2rgb(image)
    if self.transform:
      image = self.transform(image)
    return image

  def get_recipe_human (self, idx):
    "human readable form of the recipe"
    d = self.raw_data[idx]
    materials = {}
    if 'materialComponents' in d:
      for material in d['materialComponents']:
        name = material['material']['name']
        percentage = material['percentageAmount']
        materials[name] = percentage
    return materials

  def __len__ (self):
    return len(self.raw_data)

  def __getitem__ (self, idx):
    sample = {
      'image': self.get_image(idx),
      'recipe': self.get_recipe(idx),
      'recipe_human': json.dumps(self.get_recipe_human(idx))
    }
    return sample

# def view_dataset ():
#   glaze_dataset = GlazyDataset('./data.json', 'images')
# 
#   for i in range(len(glaze_dataset)):
#     sample = glaze_dataset[i]
#     print(sample)
#     if i == 3:
#       break
# 
# view_dataset()