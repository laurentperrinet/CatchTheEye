#!/usr/bin/env python3
# -*- coding: utf-8 -*

from __future__ import print_function
# import time
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.multiprocessing as mp

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
        # GPU boilerplate
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        if self.args.verbose:
            print('cuda?', self.args.cuda)

        if self.args.cuda:
            self.model.cuda()
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)
        # DATA
        self.d = Data(self.args)
        self.model = Net(args).to(self.device)
        # MODEL
        self.optimizer = optim.SGD(self.model.parameters(),
                                    lr=self.args.lr, momentum=self.args.momentum)

    def train(self, rank=0):
        torch.manual_seed(self.args.seed + rank)
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            #test_epoch(model, test_loader)

    def train_epoch(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.d.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if self.args.log_interval>0:
                if batch_idx % self.args.log_interval == 0:
                    print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_idx * len(data), len(self.d.train_loader.dataset),
                        100. * batch_idx / len(self.d.train_loader), loss.item()))


    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.d.test_loader:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(self.d.test_loader.dataset)

        if args.log_interval>0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.d.test_loader.dataset),
            100. * correct / len(self.d.test_loader.dataset)))
        return correct.numpy() / len(self.d.test_loader.dataset)

    def protocol(self):

        if self.args.num_processes>1:
            self.model.share_memory() # gradients are allocated lazily, so they are not shared here

            processes = []
            for rank in range(self.args.num_processes):
                p = mp.Process(target=self.train, args=(rank,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            for epoch in range(1, self.args.epochs + 1):
                if self.args.log_interval>0: # rajout de la commande pour pouvoir print ou non les différents epoch ou juste le résultat
                    print('Train Epoch: {} '.format(epoch))
                self.train()
        Accuracy = self.test()
        return Accuracy

    def main(self):
        Accuracy = self.protocol()
        print('Test set: Final Accuracy: {:.3f}%'.format(Accuracy*100)) # print que le pourcentage de réussite final

def init_cdl(batch_size=64, test_batch_size=1000, epochs=10,
            lr=0.01, momentum=0.5, no_cuda=True, num_processes=1, seed=42,
            log_interval=10, dimension=50, verbose=False):
    # Training settings
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=test_batch_size, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=epochs, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=lr, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=momentum, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=no_cuda,
                        help='disables CUDA training')
    parser.add_argument('--num-processes', type=int, default=num_processes,
                        help='using multi-processing')
    parser.add_argument('--seed', type=int, default=seed, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=log_interval, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dimension', type=int, default=dimension, metavar='D',
                        help='the dimension of the second neuron network')
    parser.add_argument("-v", "--verbose", action="store_true", default=verbose,
                        help="increase output verbosity")
    return parser.parse_args()

def init(batch_size=64, test_batch_size=1000, epochs=10,
            lr=0.01, momentum=0.5, no_cuda=True, num_processes=1, seed=42,
            log_interval=10, dimension=50, verbose=False):
    # Training settings
    import easydict
    return easydict.EasyDict({
            "batch_size": batch_size,
            "test_batch_size": test_batch_size,
            "epochs":epochs,
            "lr": lr,
            "momentum":momentum,
            "no_cuda": no_cuda,
            "num_processes": num_processes,
            "seed": seed,
            "log_interval": log_interval,
            "dimension": dimension,
            "verbose": verbose,
    })

if __name__ == '__main__':
    args = init_cdl()

    ml = ML(args)
    ml.main()
