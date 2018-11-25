"""
basic lienar regression of the primary and secondary color to the composisiton
"""
from datasets import GlazeColor2CompositionDataset
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from utils import get_data_loaders

ds = GlazeColor2CompositionDataset()
train_ds, test_ds = get_data_loaders(ds)

reg = linear_model.LinearRegression()
train_x = [d for d, _ in train_ds]
train_y = [d.numpy() for _, d in train_ds]
print(ds[0][0])
print(ds[0][1].numpy())
print(train_ds[0][0])
print(train_ds[0][1].numpy())
print(train_y[0])

reg.fit(train_x, train_y)
