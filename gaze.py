dataset_folder = 'dataset'
dataset_faces_folder = 'dataset_faces'
batch_size = 8
no_cuda = False
test_batch_size = 1
size_test_set = .2
do_adam = False
epochs = 40
lr = 0.025
momentum = 0.05
num_processes = 1
seed = 42
log_interval = 0 # period with which we report results for the loss
fullsize = 75
crop = 64 # int(.9*fullsize)
size = 64
mean = .6
std = .3
conv1_dim = 9
conv1_kernel_size = 18
conv2_dim = 36
conv2_kernel_size = 14
bn1_momentum = .5
bn2_momentum = .5
dimension = 30
verbose = False
stride1 = 2
stride2 = 4
N_cv = 8
# DEBUG
# epochs = 2
# N_cv = 2

import easydict
def init(dataset_folder=dataset_folder, dataset_faces_folder=dataset_faces_folder, batch_size=batch_size, test_batch_size=test_batch_size, size_test_set=size_test_set, epochs=epochs,
            do_adam=do_adam, lr=lr, momentum=momentum, no_cuda=no_cuda, num_processes=num_processes, seed=seed,
            log_interval=log_interval, fullsize=fullsize, crop=crop, size=size, mean=mean, std=std,
            conv1_dim=conv1_dim, conv1_kernel_size=conv1_kernel_size,
            conv2_dim=conv2_dim, conv2_kernel_size=conv2_kernel_size,
            bn1_momentum=bn1_momentum, bn2_momentum=bn2_momentum,
            stride1=stride1, stride2=stride2, N_cv=N_cv,
            dimension=dimension, verbose=verbose):
    # Training settings
    kwargs = {}
    kwargs.update(dataset_folder=dataset_folder, dataset_faces_folder=dataset_faces_folder, batch_size=batch_size, test_batch_size=test_batch_size, size_test_set=size_test_set, epochs=epochs,
                do_adam=do_adam, lr=lr, momentum=momentum, no_cuda=no_cuda, num_processes=num_processes, seed=seed,
                log_interval=log_interval, fullsize=fullsize, crop=crop, size=size, mean=mean, std=std,
                conv1_dim=conv1_dim, conv1_kernel_size=conv1_kernel_size,
                conv2_dim=conv2_dim, conv2_kernel_size=conv2_kernel_size,
                bn1_momentum=bn1_momentum, bn2_momentum=bn2_momentum,
                stride1=stride1, stride2=stride2, N_cv=N_cv,
                dimension=dimension, verbose=verbose
                )
    # print(kwargs, locals())
    return easydict.EasyDict(kwargs)

import numpy as np

class FaceExtractor:
    def __init__(self):

        import dlib
        self.detector = dlib.get_frontal_face_detector()

    def get_bbox(self, frame, do_center=True):
        N_X, N_Y, three = frame.shape
        dets = self.detector(frame, 1)
        bbox = dets[0]
        t, b, l, r = bbox.top(), bbox.bottom(), bbox.left(), bbox.right()
        if do_center:
            height = b - t
            # print(height, N_Y//2 - height, N_Y//2 + height)
            l = np.max((N_Y//2 - height, 0))
            r = np.min((N_Y//2 + height, N_Y))
            #TODO make a warning if we get out of the window?
            return t, b, l, r
        else:
            return t, b, l, r

    def face_extractor(self, frame, bbox=None):
        if bbox is None:
            t, b, l, r = self.get_bbox(frame)
        else:
            t, b, l, r = bbox
        face = frame[t:b, l:r, :]
        return face

import os
import torch
torch.set_default_tensor_type('torch.FloatTensor')
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torchvision
import torch.optim as optim

class Data:
    def __init__(self, args):
        self.args = args

        # making sure that all folders exist
        try:
            os.mkdir(self.args.dataset_faces_folder)
        except:
            pass
        self.classes = ['blink', 'center', 'left', 'right'] # TODO : get from the full dataset
        for label in self.classes:
            try:
                os.mkdir(os.path.join(self.args.dataset_faces_folder, label))
                print('Creating folder ', os.path.join(self.args.dataset_faces_folder, label))
            except:
                pass

        # GPU boilerplate
        if self.args.verbose:
            if not self.args.no_cuda and not torch.cuda.is_available():
                print('Trying to load cuda, but it is not available')
        self.args.no_cuda = self.args.no_cuda or not torch.cuda.is_available()
        #if self.args.verbose:
        #    print('no cuda?', self.args.no_cuda)
        kwargs = {'num_workers': 1, 'pin_memory': True} if not args.no_cuda else {'num_workers': 1}
        # https://pytorch.org/docs/master/torchvision/transforms.html#torchvision.transforms.Resize
        # Resize the input PIL Image to the given size.
        tr = transforms.Resize((args.fullsize, 2*args.fullsize))
        tcc = transforms.CenterCrop((args.crop, 2*args.crop))
        tr2 = transforms.Resize((args.size, 1*args.size))
        ttt= transforms.ToTensor()
        tn = transforms.Normalize(mean=[args.mean]*3, std=[args.std]*3)

        self.train_transform = transforms.Compose([
            tr,
            # https://pytorch.org/docs/master/torchvision/transforms.html#torchvision.transforms.RandomAffine
            #transforms.RandomAffine(degrees=5, scale=(.9, 1.1), shear=3, resample=False, fillcolor=0),
            transforms.RandomAffine(degrees=2.5, shear=1., resample=False, fillcolor=0),
            tcc, tr2, ttt, tn,
            ])

        self.test_transform = transforms.Compose([
            tr,
            tcc, tr2, ttt, tn,
            ])
        try:
            self.dataset = ImageFolder(self.args.dataset_faces_folder, self.train_transform)
            #self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
            #self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=1)

            # https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
            num_train = len(self.dataset)
            # indices = list(range(num_train))
            split = int(np.floor(self.args.size_test_set * num_train))
            if self.args.verbose:
                print('Found', num_train, 'sample images; ', num_train-split, ' to train', split, 'to test')

            # N-batch_size, C-num_channels , H-height, W-width
            from torch.utils.data import random_split
            train_dataset, test_dataset = random_split(self.dataset, [num_train-split, split])

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.test_batch_size, **kwargs)
            self.classes = self.dataset.classes #'blink', 'left ', 'right', ' fix '
        except Exception as e:
            print('Could not load dataset', e)

    def show(self, noise_level=.4, nrow=8, transpose=True):

        images, foo = next(iter(self.train_loader))
        # https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid
        from torchvision.utils import make_grid
        npimg = make_grid(images, normalize=True, nrow=nrow).numpy()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=((8, 5)))
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
        self.args = args
        # data is in the format (N, C, H, W)
        self.conv1 = nn.Conv2d(3, args.conv1_dim, kernel_size=args.conv1_kernel_size)
        padding1 = args.conv1_kernel_size - 1 # total padding in layer 1 (before max pooling)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d
        out_height_1 = (args.size - padding1 - args.stride1) // args.stride1 + 1
        out_width_1 = (1*args.size - padding1 - args.stride1) // args.stride1 + 1
        # TODO : self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(args.conv1_dim, args.conv2_dim, kernel_size=args.conv2_kernel_size)
        padding2 = args.conv2_kernel_size - 1 # total padding in layer 2
        out_height_2 = (out_height_1 - padding2 - args.stride2) // args.stride2 + 1
        out_width_2 = (1*out_width_1 - padding2 - args.stride2) // args.stride2 + 1
        fc1_dim = (out_width_2*out_height_2) * args.conv2_dim
        self.fc1 = nn.Linear(fc1_dim, args.dimension)
        self.bn1 = nn.BatchNorm1d(args.dimension, momentum=args.bn1_momentum)
        self.fc2 = nn.Linear(args.dimension, len(self.args.classes))
        self.bn2 = nn.BatchNorm1d(len(self.args.classes), momentum=args.bn2_momentum)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=[self.args.stride1, self.args.stride1]))#, stride=[self.args.stride1, self.args.stride1]))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=[self.args.stride2, self.args.stride2]))#, stride=[self.args.stride2, self.args.stride2]))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=[self.args.stride2, self.args.stride2]))#, stride=[self.args.stride2, self.args.stride2]))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = self.bn2(self.fc2(x))
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


class ML():
    def __init__(self, args):
        self.args = args
        # GPU boilerplate
        self.args.no_cuda = self.args.no_cuda or not torch.cuda.is_available()
        # if self.args.verbose: print('cuda?', not self.args.no_cuda)
        self.device = torch.device("cpu" if self.args.no_cuda else "cuda")
        torch.manual_seed(self.args.seed)
        # DATA
        self.dataset = Data(self.args)
        self.args.classes = self.dataset.classes
        # MODEL
        self.model = Net(self.args).to(self.device)
        if not self.args.no_cuda:
            # print('doing cuda')
            torch.cuda.manual_seed(self.args.seed)
            self.model.cuda()

        if self.args.do_adam:
            # see https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864
            self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.args.lr, betas=(1.-self.args.momentum, 0.999), eps=1e-8)
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                    lr=self.args.lr, momentum=self.args.momentum)
    # def forward(self, img):
    #     # normalize img
    #     return (img - self.mean) / self.std

    def train(self, path=None, seed=None):
        if not path is None:
            # using a data_cache
            if os.path.isfile(path):
                self.model.load_state_dict(torch.load(path))
                print('Loading file', path)
            else:
                print('Training model...')
                self.train(path=None, seed=seed)
                torch.save(self.model.state_dict(), path) #save the neural network state
                print('Model saved at', path)
        else:
            # cosmetics
            try:
                from tqdm import tqdm
                #from tqdm import tqdm_notebook as tqdm
                verbose = 1
            except ImportError:
                verbose = 0
            if self.args.verbose == 0 or verbose == 0:
                def tqdm(x, desc=None):
                    if desc is not None: print(desc)
                    return x

            # setting up training
            if seed is None:
                seed = self.args.seed
            self.model.train(True)
            for epoch in tqdm(range(1, self.args.epochs + 1), desc='Train Epoch' if self.args.verbose else None):
                loss = self.train_epoch(epoch, seed, rank=0)
                # report classification results
                if self.args.verbose and self.args.log_interval>0:
                    if epoch % self.args.log_interval == 0:
                        status_str = '\tTrain Epoch: {} \t Loss: {:.6f}'.format(epoch, loss)
                        try:
                            from tqdm import tqdm
                            tqdm.write(status_str)
                        except Exception as e:
                            print(e)
                            print(status_str)
            self.model.train(False)

    def train_epoch(self, epoch, seed, rank=0):
        torch.manual_seed(seed + epoch + rank*self.args.epochs)
        for batch_idx, (data, target) in enumerate(self.dataset.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            # Clear all accumulated gradients
            self.optimizer.zero_grad()
            # Predict classes using images from the train set
            output = self.model(data)
            # Compute the loss based on the predictions and actual labels
            loss = F.nll_loss(output, target)
            # Backpropagate the loss
            loss.backward()
            # Adjust parameters according to the computed gradients
            self.optimizer.step()
        return loss.item()

    def classify(self, image, t):
        from PIL import Image
        image = Image.fromarray(image)#.astype('uint8'), 'RGB')
        data = t(image).unsqueeze_(0)
        # data.requires_grad = False
        self.model.eval()
        output = self.model(data)
        return np.exp(output.data.numpy()[0, :].astype(np.float))

    def test(self, dataloader=None):
        if dataloader is None:
            dataloader = self.dataset.test_loader
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, reduction='elementwise_mean').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(self.dataset.test_loader.dataset)

        if self.args.log_interval>0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.dataset.test_loader.dataset),
            100. * correct / len(self.dataset.test_loader.dataset)))
        return correct.numpy() / len(self.dataset.test_loader.dataset)

    def show(self, gamma=.5, noise_level=.4, transpose=True, only_wrong=False):
        for idx, (data, target) in enumerate(self.dataset.test_loader):
            #data, target = next(iter(self.dataset.test_loader))
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            if only_wrong and not pred == target:
                #print(target, self.dataset.dataset.imgs[self.dataset.test_loader.dataset.indices[idx]])
                print('target:' + ' '.join('%5s' % self.dataset.dataset.classes[j] for j in target))
                print('pred  :' + ' '.join('%5s' % self.dataset.dataset.classes[j] for j in pred))
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


    def main(self, path=None, seed=None):
        self.train(path=path, seed=seed)
        Accuracy = self.test()
        return Accuracy


import time
class MetaML:
    def __init__(self, args, base = 2, N_scan = 9, tag='', verbose=0, log_interval=0):
        self.args = args
        self.seed = args.seed

        self.base = base
        self.N_scan = N_scan
        self.tag = tag
        self.default = dict(verbose=verbose, log_interval=log_interval)

    def test(self, args, seed):
        # makes a loop for the cross-validation of results
        Accuracy = []
        for i_cv in range(self.args.N_cv):
            ml = ML(args)
            ml.train(seed=seed + i_cv)
            Accuracy.append(ml.test())
        return np.array(Accuracy)

    def protocol(self, args, seed):
        t0 = time.time()
        Accuracy = self.test(args, seed)
        t1 = time.time() - t0
        Accuracy = np.hstack((Accuracy, [t1]))
        return Accuracy

    def scan(self, parameter, values):
        import os
        try:
            os.mkdir('_tmp_scanning')
        except:
            pass
        print('scanning over', parameter, '=', values)
        seed = self.seed
        Accuracy = {}
        for value in values:
            if isinstance(value, int):
                value_str = str(value)
            else:
                value_str = '%.3f' % value
            path = '_tmp_scanning/' + parameter + '_' + self.tag + '_' + value_str.replace('.', '_') + '.npy'
            print ('For parameter', parameter, '=', value_str, ', ', end=" ")
            if not(os.path.isfile(path + '_lock')):
                if not(os.path.isfile(path)):
                    open(path + '_lock', 'w').close()
                    try:
                        args = easydict.EasyDict(self.args.copy())
                        args[parameter] = value
                        Accuracy[value] = self.protocol(args, seed)
                        np.save(path, Accuracy[value])
                        os.remove(path + '_lock')
                    except Exception as e:
                        print('Failed with error', e)
                else:
                    Accuracy[value] = np.load(path)

                try:
                    print('Accuracy={:.1f}% +/- {:.1f}%'.format(Accuracy[value][:-1].mean()*100, Accuracy[value][:-1].std()*100),
                  ' in {:.1f} seconds'.format(Accuracy[value][-1]))
                except Exception as e:
                    print('Failed with error', e)

            else:
                print(' currently locked with ', path + '_lock')
            seed += 1
        return Accuracy

    def parameter_scan(self, parameter, display=False):
        values = self.args[parameter] * np.logspace(-1, 1, self.N_scan, base=self.base)
        if isinstance(self.args[parameter], int):
            # print('integer detected') # DEBUG
            values =  [int(k) for k in values]
        Accuracy = self.scan(parameter, values)
        if display:
            fig, ax = plt.subplots(figsize=(8, 5))



        return Accuracy


if __name__ == '__main__':
    import os
    filename = 'figures/accuracy.pdf'
    if not os.path.exists(filename) :
        args = init(verbose=0, log_interval=0, epochs=20)
        from gaze import MetaML
        mml = MetaML(args)
        Accuracy = mml.protocol(args, 42)
        print('Accuracy', Accuracy[:-1].mean(), '+/-', Accuracy[:-1].std())
        import numpy as np
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=((8, 5)))
        n, bins, patches = ax.hist(Accuracy[:-1]*100, bins=np.linspace(0, 100, 100), alpha=.4)
        ax.vlines(np.median(Accuracy[:-1])*100, 0, n.max(), 'g', linestyles='dashed', label='median')
        ax.vlines(25, 0, n.max(), 'r', linestyles='dashed', label='chance level')
        ax.vlines(100, 0, n.max(), 'k', label='max')
        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('Smarts')
        ax.legend(loc='best')
        plt.show()
        plt.savefig(filename)
        plt.savefig(filename.replace('.pdf', '.png'))

    print(50*'-')
    print(' parameter scan')
    print(50*'-')

    if False :
        print(50*'-')
        print('Default parameters')
        print(50*'-')
        args = init(verbose=0, log_interval=0)
        ml = ML(args)
        ml.main()
    if False :
        args = init(verbose=0, log_interval=0)
        mml = MetaML(args)
        if torch.cuda.is_available():
            mml.scan('no_cuda', [True, False])
        else:
            mml.scan('no_cuda', [True])

    for base in [2]:#, 8]:
        print(50*'-')
        print(' base=', base)
        print(50*'-')

        print(50*'-')
        print(' parameter scan : data')
        print(50*'-')
        args = init(verbose=0, log_interval=0)
        mml = MetaML(args, base=base)
        for parameter in ['size', 'fullsize', 'crop', 'mean', 'std']:
            mml.parameter_scan(parameter)

        print(50*'-')
        args = init(verbose=0, log_interval=0)
        mml = MetaML(args)
        print(' parameter scan : network')
        print(50*'-')
        for parameter in ['conv1_kernel_size',
                          'conv1_dim',
                          'bn1_momentum',
                          'conv2_kernel_size',
                          'conv2_dim',
                          'bn2_momentum',
                          'stride1', 'stride2',
                          'dimension']:
            mml.parameter_scan(parameter)

        args = init(verbose=0, log_interval=0)
        mml = MetaML(args, base=base)
        print(' parameter scan : learning ')
        print(50*'-')
        print('Using SGD')
        print(50*'-')
        for parameter in ['lr', 'momentum', 'batch_size', 'epochs']:
            mml.parameter_scan(parameter)
        print(50*'-')
        print('Using ADAM')
        print(50*'-')
        args = init(verbose=0, log_interval=0, do_adam=True)
        mml = MetaML(args, tag='adam')
        for parameter in ['lr', 'momentum', 'batch_size', 'epochs']:
            mml.parameter_scan(parameter)
