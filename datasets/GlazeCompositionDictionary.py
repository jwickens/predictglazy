class CompositionDictionary:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.process_compositions()

    def process_materials(self):
        self.by_mid = {}
        self.id_to_mid = {}
        i = 0
        for data in self.raw_data:
            if 'materialComponents' in data:
                for material in data['materialComponents']:
                    mid = material['material']['id']
                    name = material['material']['name']
                    if mid in self.by_mid:
                        if self.by_mid[mid]['name'] != name:
                            print('id mismatch for material %s' % name)
                    else:
                        self.by_mid[mid] = {'name': name, 'norm_id': i}
                        self.id_to_mid[i] = mid
                        i += 1
            else:
                print('no material components in %s' % data['id'])
        self.size = i + 1

    def get_material_by_id(self, idx):
        mid = self.id_to_mid[idx]
        return self.by_mid[mid]

    def __len__(self):
        return self.size