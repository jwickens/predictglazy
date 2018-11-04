import json
import torchvision.transforms as transforms
from skimage.color import rgba2rgb
from skimage import io, transform


def load_raw_data(json_file="data.json"):
    with open(json_file) as f:
        return json.load(f)


transform_norm = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def load_image_as_vec(image_path):
    image = io.imread(image_path)
    image = transform.resize(
        image, (32, 32), mode='constant', anti_aliasing=True)
    if image.shape[2] == 4:
        image = rgba2rgb(image)
    image = transform_norm(image).float()
    return image
