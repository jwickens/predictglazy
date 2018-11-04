import json
import torch
import torch.nn as nn
import torch.optim as optim
import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from GlazeDataset import GlazeDataset
from GlazeRecipes import GlazeRecipes
from GlazeMaterialDictionary import MaterialDictionary
from GlazeNet1 import Net


def load_raw_data(json_file="data.json"):
    with open(json_file) as f:
        return json.load(f)


def get_data_loaders(material_dict, raw_data):
    recipes = GlazeRecipes(raw_data, material_dict)
    full_dataset = GlazeDataset(material_dict, raw_data, recipes)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=True, num_workers=2)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2)
    return trainloader, testloader


def train():
    raw_data = load_raw_data()
    material_dict = MaterialDictionary(raw_data)
    model = Net(len(material_dict))
    train_loader, val_loader = get_data_loaders(material_dict, raw_data)
    loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trainer = create_supervised_trainer(model, optimizer, loss)
    metrics = {'L2': ignite.metrics.MeanPairwiseDistance(
    )}
    evaluator = create_supervised_evaluator(model, metrics)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.2f}".format(
            trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg L2: {:.2f}"
              .format(trainer.state.epoch, metrics['L2']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg L2: {:.2f}"
              .format(trainer.state.epoch, metrics['L2']))

    trainer.run(train_loader, max_epochs=10)
    return model
