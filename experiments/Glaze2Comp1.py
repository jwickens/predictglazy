"""
Given an image of a glaze predict the chemical composition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from datasets import GlazeCompositionDataset
from utils import get_data_loaders
from torch import autograd


class Net(nn.Module):
    def __init__(self, out_D):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 27)

        # self.conv1 = nn.Conv2d(3, 3, 3)
        # self.conv2 = nn.Conv2d(3, 1, 3)
        # self.fc1 = nn.Linear(36, out_D)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, 36)
        # x = self.fc1(x)
        # return x


class LogCosLoss (torch.nn.Module):
    def forward(self, x, y):
        return torch.log(torch.cosh(y - x)).sum()


def train():
    ds = GlazeCompositionDataset()
    train_loader, val_loader = get_data_loaders(ds)
    out_D = len(ds.compounds)
    print('Out Dimension is %i', out_D)
    model = Net(out_D)
    loss = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.5)
    trainer = create_supervised_trainer(model, optimizer, loss)
    metrics = {'MSE': ignite.metrics.RootMeanSquaredError(
    )}
    evaluator = create_supervised_evaluator(model, metrics)
    saver = ignite.handlers.ModelCheckpoint('./checkpoints/models', 'chkpoint',
                                            save_interval=2, n_saved=4, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              saver, {'glaze_net_3': model})
    print(model.state_dict().keys())

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        iter = (trainer.state.iteration - 1) % len(train_loader) + 1
        if iter % 10 == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.10f}".format(
                trainer.state.epoch, iter, len(train_loader), trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  MSE: {:.2f}"
              .format(trainer.state.epoch, metrics['MSE']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  MSE: {:.2f}"
              .format(trainer.state.epoch, metrics['MSE']))

    with autograd.detect_anomaly():
        trainer.run(train_loader, max_epochs=100)
    return model


if __name__ == '__main__':
    train()
