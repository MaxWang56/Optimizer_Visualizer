from CustomeModels import ModelThreeLinearDropout
from CustomOptimizers import CosineRateDecay

import torch
from torch import optim

import torchvision
from torchvision import transforms

import torch.nn.functional as F

from torch.utils import data

import matplotlib.pyplot as plt

n_epochs = 20
batch_size_train = 256
batch_size_test = 1000
learning_rate = 0.1
momentum = 0.5
wd_decay = 0.001
torch.backends.cudnn.enabled = False

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST(
    '/files', train=True, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size_train, shuffle=True)
mnist_test = torchvision.datasets.MNIST(
    'files', train=False, transform=trans, download=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size_test, shuffle=True)

# getting training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

network = ModelThreeLinearDropout()
optimizer = CosineRateDecay(network.parameters(), learning_rate, learning_rate * 2, 10, weight_decay=0.001)


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step(epoch)


test_performance = []


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_performance.append(correct / len(test_loader.dataset))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, n_epochs + 1):
    print(epoch)
    train(epoch)
    test()

x = [i for i in range(n_epochs)]
plt.plot(x, test_performance)

plt.xlabel('Epochs')
plt.ylabel('Test Performance')
plt.show(block=True)
plt.interactive(False)
