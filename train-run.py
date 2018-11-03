import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from GlazyDataset import GlazyDataset
from GlazeNet1 import Net
from utils import humanize_output

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

full_dataset = GlazyDataset(transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

net = Net(full_dataset.out_dim)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train ():
  for epoch in range(2):  # loop over the dataset multiple times
      print(trainloader)
      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
          inputs, labels = data['image'].float(), data['recipe']

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if i % 100 == 99:    # print every 100 mini-batches
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
              running_loss = 0.0
  print('Finished Training')

def evaluate ():
  pdist = nn.PairwiseDistance(p=2)
  sum_dist = 0
  with torch.no_grad():
    for data in testloader:
      inputs, labels, human_labels = data['image'].float(), data['recipe'], data['recipe_human']
      outputs = net(inputs)
      dist = pdist(outputs, labels)
      sum_dist += dist
      for i, o in enumerate(outputs):
        print('---------------')
        print('L2: %s' % dist)
        print('recieved:')
        print(humanize_output(full_dataset, o))
        print('expected:')
        print(human_labels[i])
        print('---------------')
  print('total L2 %s' % sum_dist)
  
train()
evaluate()