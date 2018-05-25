from __future__ import print_function
# import time
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse

class Data():
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        print(args.dimension)
        self.fc1 = nn.Linear(320, args.dimension)
        self.fc2 = nn.Linear(args.dimension, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ML():
    def __init__(self, args):

        self.args = args
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        print ('cuda?', self.args.cuda)

        if self.args.cuda:
            self.model.cuda()

        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        self.d = Data(self.args)
        self.model = Net(args)
        self.optimizer = optim.SGD(self.model.parameters(),
                                    lr=self.args.lr, momentum=self.args.momentum)

    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.d.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if self.args.log_interval>0: # rajout de la commande pour pouvoir print ou non les différents epoch ou juste le résultat
                if batch_idx % args.log_interval == 0:
                    print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_idx * len(data), len(self.d.train_loader.dataset),
                        100. * batch_idx / len(self.d.train_loader), loss.data[0]))

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.d.test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(self.d.test_loader.dataset)

        if args.log_interval>0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.d.test_loader.dataset),
            100. * correct / len(self.d.test_loader.dataset)))
        print('correct', correct.numpy())
        return correct.numpy() / len(self.d.test_loader.dataset)

    def protocol(self):
        for epoch in range(1, self.args.epochs + 1):
            if self.args.log_interval>0: # rajout de la commande pour pouvoir print ou non les différents epoch ou juste le résultat
                print('Train Epoch: {} '.format(epoch))
            self.train()
        Accuracy = self.test()
        print('Accuracy', Accuracy)
        return Accuracy

    def main(self):
        Accuracy = self.protocol()
        print('Test set: Final Accuracy: {:.3f}%'.format(Accuracy*100)) # print que le pourcentage de réussite final

def init(epochs=10, lr=0.01, momentum=0.5, cuda=False, seed=42,
            log_interval=10, dimension=50):
    # Training settings
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=momentum, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dimension', type=int, default = 50, metavar='D',
                        help='the dimension of the second neuron network')
    parser.add_argument("-v", "--verbose", action="store_true", default=True,
                        help="increase output verbosity")
    return parser.parse_args()

if __name__ == '__main__':
    args = init()

    ml = ML(args)
    ml.main()
