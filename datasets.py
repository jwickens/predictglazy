
import os
import json
from torch.utils.data import Dataset, DataLoader
import torch
import math
import numpy as np
from PIL import Image
from utils import load_image_as_vec, tensor_as_perfect_100_total, hex2rgb, center_crop_PIL

CALCULATED_COMPOUND_ATTRS = ['SiO2Al2O3Ratio', 'R2OTotal', 'ROTotal']


class GlazeBaseDataset (Dataset):
    """
    Base dataset for working with glazey data
    """

    def __init__(self, image_dir=os.path.join(os.path.dirname(__file__), './data/images'), augment_n=1):
        self.image_dir = image_dir
        self.load_raw_data()
        self.material_by_mid = {}
        self.materials = []
        self.compounds = []
        self.compounds_by_name = {}
        self.augment_n = augment_n

    def load_raw_data(self, json_file=os.path.join(os.path.dirname(__file__), './data/glaze_data.json')):
        with open(json_file) as f:
            self.raw_data = json.load(f)

    def index(self, funcs):
        for i in range(len(self.raw_data)):
            for f in funcs:
                f(i)

    def raw_index(self, idx):
        return math.floor(idx / self.augment_n)

    def index_material(self, idx):
        raw_materials = self.get_materials_raw(idx)
        for material in raw_materials:
            mid = material['material']['id']
            name = material['material']['name']
            if mid in self.material_by_mid:
                if self.material_by_mid[mid]['name'] != name:
                    print('id mismatch for material %s' % name)
                else:
                    self.material_by_mid[mid] = {
                        'name': name, 'norm_id': len(self.materials)}
                    self.materials.append(mid)

    def index_composition(self, idx):
        composition = self.get_composition_raw(idx)
        for compound, _ in composition.items():
            if compound in self.compounds_by_name:
                pass
            elif compound in CALCULATED_COMPOUND_ATTRS:
                pass
            else:
                self.compounds_by_name[compound] = len(self.compounds)
                self.compounds.append(compound)

    def get_image_name(self, idx):
        d = self.raw_data[self.raw_index(idx)]
        image_name = os.path.join(
            self.image_dir, d['selectedImage']['filename'])
        return image_name

    def get_image(self, idx):
        return load_image_as_vec(self.get_image_name(idx))

    def get_materials_raw(self, idx):
        d = self.raw_data[idx]
        if 'materialComponents' in d:
            return d['materialComponents']
        else:
            print('no material components in %s' % d['id'])
            return {}

    def get_composition_raw(self, idx):
        d = self.raw_data[idx]
        return d['analysis']['umfAnalysis']

    def get_composition(self, idx):
        d = self.get_composition_raw(self.raw_index(idx))
        arr = [float(d[compound])
               if compound in d else 0.0 for compound in self.compounds]
        t = torch.FloatTensor(arr)
        return t

    def get_color_raw(self, idx):
        d = self.raw_data[idx]
        image = d['selectedImage']
        primary, secondary = '00000', '000000'
        if 'dominantHexColor' in image:
            primary = image['dominantHexColor']
        else:
            print('No dominant hex color for %i' % (d['id']))
        if 'secondaryHexColor' in image:
            secondary = image['secondaryHexColor']
        else:
            print('No secondary hex color for %i' % (d['id']))
        return primary, secondary

    def get_color(self, idx):
        primary, secondary = self.get_color_raw(self.raw_index(idx))
        return [*hex2rgb(primary), *hex2rgb(secondary)]

    def get_recipe(self, idx):
        return torch.as_tensor(self.recipes.as_matrix[idx]).float()

    def compute_neighbors(self):
        self.neighbors = NearestNeighbors(
            algorithm='ball_tree', metric='manhattan').fit(self.as_matrix)

    def get_closest_recipe(self, recipe):
        "takes a recipe in vector form and returns the closest recipe"
        _, indices = self.neighbors.kneighbors(
            X=recipe.detach().numpy().reshape(1, -1), n_neighbors=1)
        return indices[0]

    def get_recipe_human(self, idx):
        "human readable form of the recipe"
        d = self.raw_data[int(idx)]
        materials = {}
        if 'materialComponents' in d:
            for material in d['materialComponents']:
                name = material['material']['name']
                percentage = material['percentageAmount']
                materials[name] = percentage
        return materials

    def humanize_output(self, output):
        "Convert recipe torch tensor into a dictionary of at most 10 ingredients"
        x = output.clamp(0)
        top_10_v, top_10_i = x.topk(10)
        top_i = []
        top_v = []
        for i, v in enumerate(top_10_v):
            if v > 0.001:
                top_i.append(top_10_i[i])
                top_v.append(v)

        if len(top_i) == 0:
            return {}

        norm_v = tensor_as_perfect_100_total(top_v)

        def get_name(idx):
            mat = self.material_dict.get_material_by_id(idx)
            return mat['name']
        result = {get_name(int(i)): int(norm_v[idx])
                  for idx, i in enumerate(top_i) if norm_v[idx] > 0}
        return result

    def __len__(self):
        return len(self.raw_data) * self.augment_n


class GlazeCompositionDataset(GlazeBaseDataset):
    """
    X is Image and Y is Composition
    """

    def __init__(self):
        super().__init__(augment_n=1000)
        self.index([self.index_composition])

    def __getitem__(self, idx):
        return self.get_image(idx), self.get_composition(idx)


class GlazeColor2CompositionDataset(GlazeBaseDataset):
    """
    X is RGB color and Y is composition
    """

    def __init__(self):
        super().__init__()
        self.index([self.index_composition])

    def __getitem__(self, idx):
        return self.get_color(idx), self.get_composition(idx)


class GlazeRecipeDataset(GlazeBaseDataset):
    """
    X is Image and Y is recipe
    """

    def __init__(self):
        super.__init__()
        self.index((self.index_material))

    def __getitem__(self, idx):
        return self.get_image(idx), self.get_recipe(idx)


class GlazeFlattenedImageDataset(GlazeBaseDataset):
    """
    Returns a flattened image only
    """

    def __getitem__(self, idx):
        image_name = self.get_image_name(idx)
        image = Image.open(image_name).convert('RGB')
        cropped = center_crop_PIL(image, 100, 100)
        flat = np.asarray(cropped).flatten()
        return flat
