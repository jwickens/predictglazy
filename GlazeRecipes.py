from sklearn.neighbors import NearestNeighbors
import numpy as np


class GlazeRecipes:
    def __init__(self, raw_data, material_dict):
        self.raw_data = raw_data
        self.material_dict = material_dict
        self.load_recipes()
        self.compute_neighbors()

    def load_recipes(self):
        self.as_matrix = []
        for d in self.raw_data:
            recipe = np.zeros(len(self.material_dict))
            if 'materialComponents' in d:
                for material in d['materialComponents']:
                    mid = material['material']['id']
                    nid = self.material_dict.by_mid[mid]['norm_id']
                    recipe[nid] = float(material['percentageAmount'])
            x = recipe / 100
            self.as_matrix.append(x)

    def compute_neighbors(self):
        self.neighbors = NearestNeighbors(
            algorithm='ball_tree', metric='minkowski').fit(self.as_matrix)

    def get_closest_recipe(self, recipe):
        "takes a recipe in vector form and returns the closest recipe"
        _, indices = self.neighbors.kneighbors(X=recipe, n_neighbors=1)
        return indices[0]

    def get_recipe_human(self, idx):
        "human readable form of the recipe"
        d = self.raw_data[idx]
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
        for i, v in enumerate(top_10_v):
            if v > 0.1:
                top_i.append(top_10_i[i])
        norm_x = tensor_as_perfect_100_total(top_i)

        def get_name(idx):
            mat = self.material_dict.get_material_by_id(idx)
            return mat['name']
        result = {get_name(int(i)): int(norm_x[i]) for i in top_i}
        return result


def tensor_as_perfect_100_total(x):
    "use lergest remainder algorithm to scale vector to percentage of 100"
    norm_x = x * 100 / x.sum()
    norm_x = norm_x.floor()
    remainder = 100 - norm_x.sum()
    _, greatest_remainders = remainder.sort(descending=False)
    j = 0
    while remainder > 0 and j < len(remainder):
        norm_x[greatest_remainders[j]] += 1
        j += 1
        remainder -= 1
        if remainder > 0 and j == len(remainder):
            j = 0
    return norm_x
