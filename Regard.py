#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import print_function

batch_size = 16
test_batch_size = 1
valid_size = .2
epochs = 100
lr = 0.02
momentum = 0.25
no_cuda = True
num_processes = 1
seed = 42
log_interval = 10
crop = 200
size = 128
mean = .35
std = .4
conv1_dim = 5
conv1_kernel_size = 5
conv2_dim = 13
conv2_kernel_size = 5
dimension = 25
verbose = False



import numpy as np
import torch
torch.set_default_tensor_type('torch.FloatTensor')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.multiprocessing as mp
import torchvision.models as models
import torchvision
# torchvision.set_image_backend('accimage')
from torch.utils.data.sampler import SubsetRandomSampler

class Data():
    def __init__(self, args):
        self.args = args
        # GPU boilerplate
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        if self.args.verbose:
            print('cuda?', self.args.cuda)
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {'num_workers': 1, 'shuffle': True}

        # self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        # self.IMAGENET_STD = [0.229, 0.224, 0.225]

        t = transforms.Compose([
            transforms.CenterCrop(args.crop),
            transforms.Resize(args.size),
            transforms.ToTensor(),
            #transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            transforms.Normalize(mean=[args.mean]*3, std=[args.std]*3),
            ])
        self.dataset = ImageFolder('dataset', t)
        #self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        #self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=1)

        # https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
        num_train = len(self.dataset)
        # indices = list(range(num_train))
        split = int(np.floor(self.args.valid_size * num_train))
        if self.args.verbose:
            print('Found', num_train, 'sample images; ', num_train-split, ' to train', split, 'to test')
        #
        # np.random.seed(self.args.seed)
        # np.random.shuffle(indices)
        #
        # train_idx, valid_idx = indices[split:], indices[:split]
        # train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)

        from torch.utils.data import random_split
        train_dataset, test_dataset = random_split(self.dataset, [num_train-split, split])

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.test_batch_size, **kwargs)

        # self.classes = self.dataset.classes #'blink', 'left ', 'right', ' fix '

    def show(self, gamma=.5, noise_level=.4, transpose=True):

        images, foo = next(iter(self.train_loader))

        from torchvision.utils import make_grid
        npimg = make_grid(images, normalize=True).numpy()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=((13, 5)))
        import numpy as np
        if transpose:
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
        else:
            ax.imshow(npimg)
        plt.setp(ax, xticks=[], yticks=[])

        return fig, ax


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, args.conv1_dim, kernel_size=args.conv1_kernel_size)
        self.conv2 = nn.Conv2d(args.conv1_dim, args.conv2_dim, kernel_size=args.conv2_kernel_size)
        #self.conv2_drop = nn.Dropout2d()
        padding = 2* (2* (args.conv1_kernel_size -1)//2 + (args.conv2_kernel_size -1)//2)
        fc1_dim = int((args.size - padding) **2 * args.conv2_dim / 16)
        self.fc1 = nn.Linear(fc1_dim, args.dimension)
        self.fc2 = nn.Linear(args.dimension, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=[2, 2], stride=[2, 2]))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=[2, 2], stride=[2, 2]))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=[2, 2], stride=[2, 2]))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# class Net(nn.Module):
#     def __init__(self, args):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         #self.pool = nn.MaxPool2d(2, 2)#
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, args.dimension)
#         self.fc2 = nn.Linear(args.dimension, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 4))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 4))
#         x = x.view(-1, 20480)
#         #x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         #x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         #x = F.dropout(x, training=self.training)
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

# cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
# cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
#
# # create a module to normalize input image so we can easily put it in a
# # nn.Sequential
# class Normalization(nn.Module):
#     def __init__(self, mean, std):
#         super(Normalization, self).__init__()
#         # .view the mean and std to make them [C x 1 x 1] so that they can
#         # directly work with image Tensor of shape [B x C x H x W].
#         # B is batch size. C is number of channels. H is height and W is width.
#         self.mean = torch.tensor(mean).view(-1, 1, 1)
#         self.std = torch.tensor(std).view(-1, 1, 1)


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
        # MODEL
        self.model = Net(args).to(self.device)
        #self.model = models.vgg19(pretrained=True).features.to(device).eval()
        self.optimizer = optim.SGD(self.model.parameters(),
                                    lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

    def train(self):
        self.model.train()

        try:
            from tqdm import tqdm_notebook as tqdm
            verbose = 1
        except ImportError:
            verbose = 0
        if self.args.verbose==0 or verbose==0:
            def tqdm(x, desc=None):
                if desc is not None: print(desc)
                return x

        for epoch in tqdm(range(1, self.args.epochs + 1), desc='Train Epoch' if self.args.verbose else None):
            self.train_epoch(epoch, rank=0)

        # if self.args.verbose :
        #     for epoch in tqdm(range(1, self.args.epochs + 1), desc='Train Epoch'):
        #         self.train_epoch(epoch, rank=0)
        # else:
        #     for epoch in range(1, self.args.epochs + 1):
        #         self.train_epoch(epoch, rank=0)

    def train_epoch(self, epoch, rank=0):
        torch.manual_seed(self.args.seed + epoch + rank*self.args.epochs)
        for batch_idx, (data, target) in enumerate(self.d.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if self.args.verbose and self.args.log_interval>0:
                if batch_idx % self.args.log_interval == 0:
                    print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.d.train_loader.dataset),
                        100. * batch_idx / len(self.d.train_loader), loss.item()))

    def test(self, dataloader=None):
        if dataloader is None:
            dataloader = self.d.test_loader
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(self.d.test_loader.dataset)

        if self.args.log_interval>0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.d.test_loader.dataset),
            100. * correct / len(self.d.test_loader.dataset)))
        return correct.numpy() / len(self.d.test_loader.dataset)

    def show(self, gamma=.5, noise_level=.4, transpose=True, only_wrong=False):

        data, target = next(iter(self.d.test_loader))
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        if only_wrong and not pred == target:
            print('target:' + ' '.join('%5s' % self.d.dataset.classes[j] for j in target))
            print('pred  :' + ' '.join('%5s' % self.d.dataset.classes[j] for j in pred))
            #print(target, pred)


            from torchvision.utils import make_grid
            npimg = make_grid(data, normalize=True).numpy()
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=((13, 5)))
            import numpy as np
            if transpose:
                ax.imshow(np.transpose(npimg, (1, 2, 0)))
            else:
                ax.imshow(npimg)
            plt.setp(ax, xticks=[], yticks=[])

            return fig, ax
        else:
            return None, None

    def protocol(self):
        # TODO: make a loop for the cross-validation of results
        self.train()
        Accuracy = self.test()
        return Accuracy

    def main(self):
        Accuracy = self.protocol()
        print('Test set: Final Accuracy: {:.3f}%'.format(Accuracy*100)) # print que le pourcentage de réussite final


def init_cdl(batch_size=batch_size, test_batch_size=test_batch_size, valid_size=valid_size, epochs=epochs,
            lr=lr, momentum=momentum, no_cuda=no_cuda, num_processes=num_processes, seed=seed,
            log_interval=log_interval, crop=crop, size=size, mean=mean, std=std,
            conv1_dim=conv1_dim, conv1_kernel_size=conv1_kernel_size,
            conv2_dim=conv2_dim, conv2_kernel_size=conv2_kernel_size,
            dimension=dimension, verbose=verbose):
    # Training settings
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=test_batch_size, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--valid-size', type=float, default=valid_size, metavar='N',
                        help='ratio of samples used for validation (default: .2)')
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
    parser.add_argument('--crop', type=int, default=crop,
                        help='size of cropped image')
    parser.add_argument('--size', type=int, default=size,
                        help='size of image')
    parser.add_argument('--conv1_dim', type=int, default=conv1_dim,
                        help='size of conv1 depth')
    parser.add_argument('--conv2_dim', type=int, default=conv2_dim,
                        help='size of conv2 depth')
    parser.add_argument('--conv1_kernel_size', type=int, default=conv1_kernel_size,
                        help='size of conv1 kernel_size')
    parser.add_argument('--conv2_kernel_size', type=int, default=conv2_kernel_size,
                        help='size of conv2 kernel_size')
    parser.add_argument('--seed', type=int, default=seed, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=log_interval, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dimension', type=int, default=dimension, metavar='D',
                        help='the dimension of the second neuron network')
    parser.add_argument("-v", "--verbose", action="store_true", default=verbose,
                        help="increase output verbosity")
    return parser.parse_args()

def init(batch_size=batch_size, test_batch_size=test_batch_size, valid_size=valid_size, epochs=epochs,
            lr=lr, momentum=momentum, no_cuda=no_cuda, num_processes=num_processes, seed=seed,
            log_interval=log_interval, crop=crop, size=size, mean=mean, std=std,
            conv1_dim=conv1_dim, conv1_kernel_size=conv1_kernel_size,
            conv2_dim=conv2_dim, conv2_kernel_size=conv2_kernel_size,
            dimension=dimension, verbose=verbose):
    # Training settings
    kwargs = {
            "batch_size": batch_size,
            "test_batch_size": test_batch_size,
            "valid_size":valid_size,
            "epochs":epochs,
            "lr": lr,
            "momentum":momentum,
            "no_cuda": no_cuda,
            "num_processes": num_processes,
            "seed": seed,
            "log_interval": log_interval,
            "dimension": dimension,
            "verbose": verbose,
    }
    kwargs.update(conv1_dim=conv1_dim, conv1_kernel_size=conv1_kernel_size,
                  conv2_dim=conv2_dim, conv2_kernel_size=conv2_kernel_size,
                  crop=crop, size=128, mean=mean, std=std
                  )
    # print(kwargs)
    import easydict
    return easydict.EasyDict(kwargs)

if __name__ == '__main__':
    args = init_cdl()
    ml = ML(args)
    ml.main()
