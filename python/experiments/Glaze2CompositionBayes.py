
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import GlazeCompositionDataset

class NN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output

net = NN(28*28*28, 1024, 10)
