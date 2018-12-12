import torchvision.transforms as transforms
from PIL import Image
import torch


transform_norm = transforms.Compose(
    [
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def load_image_as_vec(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform_norm(image).float()
    return image


def tensor_as_perfect_100_total(x):
    "use largest remainder algorithm to scale vector to percentage of 100"
    x = torch.tensor(x).float()
    norm_x = x * 100 / x.sum()
    norm_x = norm_x.floor()
    _, greatest_remainders = norm_x.sort(descending=False)
    j = 0
    remainder = 100 - norm_x.sum()
    while remainder > 0 and j < len(x):
        norm_x[greatest_remainders[j]] += 1
        j += 1
        remainder -= 1
        if remainder > 0 and j == len(x):
            j = 0
    return norm_x


def split_dataset(full_dataset):
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    print('Train size: %i, test size: %i' % (train_size, test_size))
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])
    return train_dataset, test_dataset


def get_data_loaders(full_dataset):
    train_dataset, test_dataset = split_dataset(full_dataset)
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=True, num_workers=2)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2)
    return trainloader, testloader


def hex2rgb(hex):
    """
    from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    """
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))


def center_crop_PIL(image, height, width):
    w, h = image.size   # Get dimensions
    left = (w - width)/2
    top = (h - height)/2
    right = (w + width)/2
    bottom = (h + height)/2
    cropped = image.crop((left, top, right, bottom))
    return cropped
