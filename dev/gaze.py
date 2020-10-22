# TODO: try HoG
# https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV
# hogFaceDetector = dlib.get_frontal_face_detector()
# faceRects = hogFaceDetector(frameDlibHogSmall, 0)
# for faceRect in faceRects:
#     x1 = faceRect.left()
#     y1 = faceRect.top()
#     x2 = faceRect.right()
#     y2 = faceRect.bottom()
#

# DEBUG
# epochs = 2
# N_cv = 2
#https://raw.githubusercontent.com/MorvanZhou/PyTorch-Tutorial/master/tutorial-contents/504_batch_normalization.py


def init(print_help=False, dataset_folder='dataset', dataset_faces_folder=None, # dataset_faces_folder='dataset_faces',
         dataset_train='dataset_train', dataset_test='dataset_test',
         batch_size=16, test_batch_size=16, size_test_set=.2,
         epochs=40, do_adam=False, lr=0.01, momentum=0.05,
         no_cuda=False,
         num_processes=1,
         log_interval=0,
         fullsize=75,
         crop=75,
         size=40, mean=.4, std=.3,
         conv1_dim=9,  conv1_kernel_size=8,  conv1_bn_momentum=.5,
         conv2_dim=36, conv2_kernel_size=12, conv2_bn_momentum=.5,
         dense_bn_momentum=.5,
         dimension=30,
         verbose=False, seed=42,
         stride1=2, stride2=4,
         N_cv=20,
         activation='relu') :

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder",       default=dataset_folder,       type=str,   help="Dataset Folder")
    parser.add_argument("--dataset_faces_folder", default=dataset_faces_folder, type=str,   help="Dataset Faces Folder")
    parser.add_argument("--dataset_train",        default=dataset_train,        type=str,   help="Dataset Train")
    parser.add_argument("--dataset_test",         default=dataset_test,         type=str,   help="Dataset Test")
    parser.add_argument("--batch_size",           default=batch_size,           type=int,   help="batch_size")
    parser.add_argument("--no_cuda",              default=no_cuda,              type=bool,  help="No cuda")
    parser.add_argument("--test_batch_size",      default=test_batch_size,      type=int,   help="Test Batch Size")
    parser.add_argument("--size_test_set",        default=size_test_set,        type=float, help="Size Test Set")
    parser.add_argument("--do_adam",              default=do_adam,              type=bool,  help="Do Adam")
    parser.add_argument("--epochs",               default=epochs,               type=int,   help="Epochs")
    parser.add_argument("--lr",                   default=lr,                   type=float, help="LR")
    parser.add_argument("--momentum",             default=momentum,             type=float, help="momentum")
    parser.add_argument("--num_processes",        default=num_processes,        type=int,   help="num_processes")
    parser.add_argument("--seed",                 default=seed,                 type=int,   help="seed")
    parser.add_argument("--log_interval",         default=log_interval,         type=int,   help="period with which we report results for the loss")
    parser.add_argument("--fullsize",             default=fullsize,             type=int,   help="size at the input of the transforms pipeline")
    parser.add_argument("--crop",                 default=crop,                 type=int,   help="crop") # int(.9*fullsize)
    parser.add_argument("--size",                 default=size,                 type=int,   help="size at the output of the transforms pipeline")
    parser.add_argument("--mean",                 default=mean,                 type=float, help="mean")
    parser.add_argument("--std",                  default=std,                  type=float, help="std")
    parser.add_argument("--conv1_dim",            default=conv1_dim,            type=int,   help="conv1_dim")
    parser.add_argument("--conv1_kernel_size",    default=conv1_kernel_size,    type=int,   help="conv1_kernel_size")
    parser.add_argument("--conv1_bn_momentum",    default=conv1_bn_momentum,    type=float, help="conv1_bn_momentum")
    parser.add_argument("--conv2_kernel_size",    default=conv2_kernel_size,    type=int,   help="conv2_kernel_size")
    parser.add_argument("--conv2_dim",            default=conv2_dim,            type=int,   help="conv2_dim")
    parser.add_argument("--conv2_bn_momentum",    default=conv2_bn_momentum,    type=float, help="conv2_bn_momentum")
    parser.add_argument("--dense_bn_momentum",    default=dense_bn_momentum,    type=float, help="dense_bn_momentum")
    parser.add_argument("--dimension",            default=dimension,            type=int,   help="dimension")
    parser.add_argument("--verbose",              default=verbose,              type=bool,  help="verbose")
    parser.add_argument("--stride1",              default=stride1,              type=int,   help="stride1")
    parser.add_argument("--stride2",              default=stride2,              type=int,   help="stride2")
    parser.add_argument("--N_cv",                 default=N_cv,                 type=int,   help="N_cv")
    parser.add_argument("--activation",           default=activation,           type=str,   help="'relu', 'tanh', or 'softmax'")


    if print_help :

        str_help = parser.format_help()
        str_help = str_help[str_help.find('\noptional arguments'):]

        print('Optional arguments :\n')

        for x in range(31) :
            str_help = str_help[str_help.find('\n  --')+len('\n  --'):]
            end = str_help.find(' ')
            name = str_help[:end]

            str_help = str_help[end+len(name)+1:]
            start, end = str_help.find(' '), str_help.find('\n  --')

            description = ' '
            for s in str_help[start+len(' '):end] :
                if s!=' ' or description[-1]!=' ' : description = description+s

            print('\t', name, ['\t\t\t' if len(name)<8 else '\t\t'][0] if len(name)<16  else '\t',
                  '- description --> ', description[1:], '\n\t\t\t\t',
                  '- default --> ', parser.get_default(name), sep='', end='\n\n')

    args=[]
    return parser.parse_args(args=args)

#-------------------------------------------------------------------------------------------------


import numpy as np
from LeCheapEyeTracker_copy.EyeTrackerServer import FaceExtractor
import os
import torch
torch.set_default_tensor_type('torch.FloatTensor')
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torchvision
import torch.optim as optim
import torch.nn.functional as F


class Data:

    def __init__(self, args) :

        self.args = args

        # making sure that all folders exist
        #---------------------------------------------------------------
        try :    os.mkdir(self.args.dataset_faces_folder)
        except : pass

        self.classes = ['blink', 'center', 'left', 'right'] # TODO : get from the full dataset
        for label in self.classes:
            try :
                os.mkdir(os.path.join(self.args.dataset_faces_folder, label))
                print('Creating folder ', os.path.join(self.args.dataset_faces_folder, label))
            except :
                pass

        # GPU boilerplate
        #---------------------------------------------------------------
        if self.args.verbose :
            if not self.args.no_cuda and not torch.cuda.is_available() :
                print('Trying to load cuda, but it is not available')
        self.args.no_cuda = self.args.no_cuda or not torch.cuda.is_available()
        #if self.args.verbose: print('no cuda?', self.args.no_cuda)
        kwargs = {'num_workers': 1, 'pin_memory': True} if not self.args.no_cuda else {'num_workers': 1}
        kwargs.update(drop_last=True)


        # Resize the input PIL Image to the given size.
        #           https://pytorch.org/docs/master/torchvision/transforms.html#torchvision.transforms.Resize
        #---------------------------------------------------------------
        tr  = transforms.Resize((args.fullsize, 4*args.fullsize))
        tcc = transforms.CenterCrop((args.crop, 4*args.crop))
        tr2 = transforms.Resize((args.size, 4*args.size))
        ttt = transforms.ToTensor()
        tn  = transforms.Normalize(mean=[args.mean]*3, std=[args.std]*3)

        # https://pytorch.org/docs/master/torchvision/transforms.html#torchvision.transforms.RandomAffine
        #transforms.RandomAffine(degrees=5, scale=(.9, 1.1), shear=3, resample=False, fillcolor=0),
        #transforms.RandomAffine(degrees=2.5, shear=1., resample=False, fillcolor=0),
        self.train_transform = transforms.Compose([tr, tcc, tr2, ttt, tn,])
        self.test_transform = transforms.Compose([tr, tcc, tr2, ttt, tn,])

        try :

            if self.args.dataset_faces_folder != None :
                self.dataset = ImageFolder(self.args.dataset_faces_folder, self.train_transform)

                #self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size,      shuffle=True, num_workers=1)
                #self.test_loader  = torch.utils.data.DataLoader(self.dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=1)

                # https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
                num_train = len(self.dataset)
                # indices = list(range(num_train))
                split = int(np.floor(self.args.size_test_set * num_train))
                if self.args.verbose: print('Found', num_train, 'sample images; ', num_train-split, ' to train', split, 'to test')

                # N-batch_size, C-num_channels , H-height, W-width
                from torch.utils.data import random_split
                train_dataset, test_dataset = random_split(self.dataset, [num_train-split, split])

                self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size,      **kwargs)
                self.test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=self.args.test_batch_size, **kwargs)
                self.classes = self.dataset.classes #'blink', 'left ', 'right', ' fix '


            else :
                train_dataset = ImageFolder(self.args.dataset_train, self.train_transform)
                test_dataset  = ImageFolder(self.args.dataset_test,  self.train_transform)

                if self.args.verbose: print('Found', len(train_dataset)+len(test_dataset), 'sample images; ',
                                             len(train_dataset), ' to train', len(test_dataset), 'to test')

                self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size,      **kwargs)
                self.test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=self.args.test_batch_size, **kwargs)
                self.classes = train_dataset.classes #'blink', 'left ', 'right', ' fix '

        except Exception as e :

            print('Could not load dataset', e)

    def show(self, noise_level=.4, nrow=8, transpose=True) :

        # https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid
        from torchvision.utils import make_grid
        import matplotlib.pyplot as plt
        import numpy as np

        images, foo = next(iter(self.train_loader))
        npimg = make_grid(images, normalize=True, nrow=nrow).numpy()

        fig, ax = plt.subplots(figsize=((8, 5)))
        if transpose: ax.imshow(np.transpose(npimg, (1, 2, 0)))
        else:         ax.imshow(npimg)
        plt.setp(ax, xticks=[], yticks=[])

        return fig, ax


class Net(nn.Module) :

    def __init__(self, args) :

        # TODO : rajouter la position de la tete comme entrée
        super(Net, self).__init__()
        self.args = args

        if   self.args.activation == 'relu' :    self.ACTIVATION = F.relu
        elif self.args.activation == 'tanh' :    self.ACTIVATION = torch.tanh
        elif self.args.activation == 'softmax' : self.ACTIVATION = F.softmax


        # data is in the format (N, C, H, W)
        self.conv1 = nn.Conv2d(3, args.conv1_dim, kernel_size=args.conv1_kernel_size) # TODO add , padding_mode=args.padding_mode
        self.conv1_bn = nn.BatchNorm2d(args.conv1_dim, momentum=1-args.conv1_bn_momentum)
        padding1 = args.conv1_kernel_size - 1 # total padding in layer 1 (before max pooling)

        # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d
        out_height_1 = (  args.size - padding1 - args.stride1) // args.stride1 + 1
        out_width_1  = (4*args.size - padding1 - args.stride1) // args.stride1 + 1
        # TODO : self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(args.conv1_dim, args.conv2_dim, kernel_size=args.conv2_kernel_size)
        self.conv2_bn = nn.BatchNorm2d(args.conv2_dim, momentum=1-args.conv2_bn_momentum)
        padding2 = args.conv2_kernel_size - 1 # total padding in layer 2
        out_height_2 = (out_height_1 - padding2 - args.stride2) // args.stride2 + 1
        out_width_2  = ( out_width_1 - padding2 - args.stride2) // args.stride2 + 1

        fc1_dim = (out_width_2*out_height_2) * args.conv2_dim
        self.dense_1 = nn.Linear(fc1_dim, args.dimension)
        # This momentum argument is different from one used in optimizer classes and the conventional notion of momentum. Mathematically, the update rule for running statistics here is x̂ new=(1−momentum)×x̂ +momemtum×xt, where x̂  is the estimated statistic and xt is the new observed value.
        self.dense_bn = nn.BatchNorm1d(args.dimension, momentum=1-args.dense_bn_momentum)
        self.dense_2 = nn.Linear(args.dimension, len(self.args.classes))
        self.dense_input_size = None

    def forward(self, x):

        x = self.conv1(x)
        if self.args.conv1_bn_momentum>0 : x = self.conv1_bn(x)
        x = self.ACTIVATION(F.max_pool2d(x, kernel_size=[self.args.stride1, self.args.stride1]))#, stride=[self.args.stride1, self.args.stride1]))
        # x = self.ACTIVATION(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=[self.args.stride2, self.args.stride2]))#, stride=[self.args.stride2, self.args.stride2]))
        x = self.conv2(x)
        if self.args.conv2_bn_momentum>0 : x = self.conv2_bn(x)

        x = self.ACTIVATION(F.max_pool2d(x, kernel_size=[self.args.stride2, self.args.stride2]))#, stride=[self.args.stride2, self.args.stride2]))

        if self.dense_input_size is None : self.dense_input_size= self.num_flat_features(x)
        x = x.view(-1, self.dense_input_size)
        x = self.dense_1(x)

        if self.args.dense_bn_momentum>0 : x = self.dense_bn(x)
        x = self.ACTIVATION(x)
        x = self.dense_2(x)

        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x) :

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size : num_features *= s
        return num_features


class ML():

    def __init__(self, args) :

        self.args = args
        # GPU boilerplate
        self.args.no_cuda = self.args.no_cuda or not torch.cuda.is_available()
        # if self.args.verbose: print('cuda?', not self.args.no_cuda)
        self.device = torch.device("cpu" if self.args.no_cuda else "cuda")
        torch.manual_seed(self.args.seed)

        #---------------------------------------------------------------
        # DATA
        #---------------------------------------------------------------
        self.dataset = Data(self.args)
        self.args.classes = self.dataset.classes

        #---------------------------------------------------------------
        # MODEL
        #---------------------------------------------------------------
        self.model = Net(self.args).to(self.device)
        if not self.args.no_cuda :
            # print('doing cuda')
            torch.cuda.manual_seed(self.args.seed)
            self.model.cuda()

        # see https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864
        if self.args.do_adam : self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr,
                                                          betas=(1.-self.args.momentum, 0.999), eps=1e-8)
        else :                 self.optimizer =  optim.SGD(self.model.parameters(), lr=self.args.lr,
                                                          momentum=self.args.momentum)
    # def forward(self, img):
    #     # normalize img
    #     return (img - self.mean) / self.std

    def train(self, path=None, seed=None) :

        if not path is None :

            # using a data_cache
            if os.path.isfile(path) :
                self.model.load_state_dict(torch.load(path))
                print('Loading file', path)
            else :
                print('Training model...')
                self.train(path=None, seed=seed)
                torch.save(self.model.state_dict(), path) #save the neural network state
                print('Model saved at', path)

        else :

            # cosmetics
            try :
                from tqdm import tqdm
                #from tqdm import tqdm_notebook as tqdm
                verbose = 1
            except ImportError :
                verbose = 0

            if self.args.verbose == 0 or verbose == 0 :
                def tqdm(x, desc=None) :
                    if desc is not None: print(desc)
                    return x

            # setting up training
            if seed is None: seed = self.args.seed

            self.model.train()

            for epoch in tqdm(range(1, self.args.epochs + 1), desc='Train Epoch' if self.args.verbose else None) :

                loss = self.train_epoch(epoch, seed, rank=0)
                # report classification results
                if self.args.verbose and self.args.log_interval>0 :

                    if epoch % self.args.log_interval == 0 :

                        status_str = '\tTrain Epoch: {} \t Loss: {:.6f}'.format(epoch, loss)

                        try :
                            from tqdm import tqdm
                            tqdm.write(status_str)
                        except Exception as e :
                            print(e)
                            print(status_str)

            self.model.eval()

    def train_epoch(self, epoch, seed, rank=0) :

        torch.manual_seed(seed + epoch + rank*self.args.epochs)

        for batch_idx, (data, target) in enumerate(self.dataset.train_loader) :

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()                         # Clear all accumulated gradients
            output = self.model(data)                          # Predict classes using images from the train set
            loss = F.nll_loss(output, target, reduction='sum') # Compute the loss based on the predictions and actual labels
            loss.backward()                                    # Backpropagate the loss
            self.optimizer.step()                              # Adjust parameters according to the computed gradients

        return loss.item()

    def classify(self, image, t) :

        from PIL import Image
        image = Image.fromarray(image)#.astype('uint8'), 'RGB')
        data = t(image).unsqueeze(0)
        # data.requires_grad = False
        self.model.eval()

        if not self.args.no_cuda :
            output = self.model(data.cuda())
            return np.exp(output.cpu().data.numpy()[0, :].astype(np.float))
        else :
            output = self.model(data)
            return np.exp(output.data.numpy()[0, :].astype(np.float))

    def test(self, dataloader=None) :

        if dataloader is None : dataloader = self.dataset.test_loader

        self.model.eval()
        test_loss, correct = 0, 0

        for data, target in dataloader :
            data, target = data.to(self.device), target.to(self.device)
            output       = self.model(data)
            test_loss   += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred         = output.data.max(1, keepdim=True)[1]                      # get the index of the max log-probability
            correct     += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(dataloader.dataset)

        if self.args.log_interval > 0 :
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(dataloader.dataset),
                    100.*correct / len(dataloader.dataset)))

        return correct.numpy() / len(dataloader.dataset)

    def show(self, gamma=.5, noise_level=.4, transpose=True, only_wrong=False) :

        from torchvision.utils import make_grid
        import matplotlib.pyplot as plt
        import numpy as np

        for idx, (data, target) in enumerate(self.dataset.test_loader) :

            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            pred = output.data.max(1, keepdim=False)[1] # get the index of the max log-probability

            if only_wrong and not all(pred) == all(target) :

                print('target:' + ' '.join('%5s' % self.dataset.dataset.classes[j] for j in target))
                print('pred  :' + ' '.join('%5s' % self.dataset.dataset.classes[j] for j in pred))

                npimg = make_grid(data, normalize=True).numpy()
                fig, ax = plt.subplots(figsize=((13, 5)))
                if transpose: ax.imshow(np.transpose(npimg, (1, 2, 0)))
                else:         ax.imshow(npimg)
                plt.setp(ax, xticks=[], yticks=[])

                return fig, ax

            else:

                return None, None


    def main(self, path=None, seed=None) :

        self.train(path=path, seed=seed)
        Accuracy = self.test()

        return Accuracy


import time
class MetaML :

    def __init__(self, args, base=2, N_scan=9, tag='', verbose=0, log_interval=0) :

        self.args = args
        self.seed = args.seed

        self.base    = base
        self.N_scan  = N_scan
        self.tag     = tag
        self.default = dict(verbose=verbose, log_interval=log_interval)

    def test(self, args, seed) :

        # makes a loop for the cross-validation of results
        Accuracy = []
        for i_cv in range(self.args.N_cv) :
            ml = ML(args)
            ml.train(seed=seed + i_cv)
            Accuracy.append(ml.test())
        return np.array(Accuracy)

    def protocol(self, args, seed) :

        t0 = time.time()
        Accuracy = self.test(args, seed)
        t1 = time.time() - t0
        Accuracy = np.hstack((Accuracy, [t1]))
        return Accuracy

    def scan(self, parameter, values) :

        import os
        try :    os.mkdir('_tmp_scanning')
        except : pass

        print('scanning over', parameter, '=', values)

        seed = self.seed
        Accuracy = {}

        for value in values :

            if isinstance(value, int) : value_str = str(value)
            else :                      value_str = '%.3f' % value

            path = '_tmp_scanning/' + parameter + '_' + self.tag + '_' + value_str.replace('.', '_') + '.npy'

            print ('For parameter', parameter, '=', value_str, ', ', end=" ")

            if not(os.path.isfile(path + '_lock')) :

                if not(os.path.isfile(path)) :

                    open(path + '_lock', 'w').close()
                    try :
                        args = easydict.EasyDict(self.args.copy())
                        args[parameter] = value
                        Accuracy[value] = self.protocol(args, seed)
                        np.save(path, Accuracy[value])
                        os.remove(path + '_lock')
                    except Exception as e :
                        print('Failed with error', e)

                else :
                    Accuracy[value] = np.load(path)

                try :
                    print('Accuracy={:.1f}% +/- {:.1f}%'.format(Accuracy[value][:-1].mean()*100, Accuracy[value][:-1].std()*100),
                           ' in {:.1f} seconds'.format(Accuracy[value][-1]))
                except Exception as e :
                    print('Failed with error', e)

            else :
                print(' currently locked with ', path + '_lock')

            seed += 1

        return Accuracy

    def parameter_scan(self, parameter, display=False) :

        if parameter in ['momentum', 'conv1_bn_momentum', 'conv2_bn_momentum', 'dense_bn_momentum'] :
            values = np.linspace(0, 1, self.N_scan, endpoint=True)
        else :
            values = self.args[parameter] * np.logspace(-1, 1, self.N_scan, base=self.base)

        if isinstance(self.args[parameter], int) :
            values =  [int(k) for k in values]

        Accuracy = self.scan(parameter, values)

        if display : fig, ax = plt.subplots(figsize=(8, 5))

        return Accuracy


if __name__ == '__main__' :

    import os
    filename = 'figures/accuracy.pdf'

    if False : #not os.path.exists(filename) :

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

    print(50*'-', 'parameter scan', 50*'-', sep='\n')

    if False :

        print(50*'-', 'Default parameters', 50*'-', sep='\n')
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

    for base in [2, 8]:

        print(50*'-', 'base = %s'%base, 50*'-', sep='\n')
        print(50*'-', 'parameter scan : data', 50*'-', sep='\n')

        args = init(verbose=0, log_interval=0)
        mml = MetaML(args, base=base)

        for parameter in ['size', 'fullsize', 'crop', 'mean', 'std']:
            mml.parameter_scan(parameter)

        args = init(verbose=0, log_interval=0)
        mml = MetaML(args)
        print(50*'-', ' parameter scan : network', 50*'-', sep='\n')

        for parameter in ['conv1_kernel_size', 'conv1_dim', 'conv1_bn_momentum',
                          'conv2_kernel_size', 'conv2_dim', 'conv2_bn_momentum',
                          'stride1', 'stride2', 'dense_bn_momentum', 'dimension'] :
            mml.parameter_scan(parameter)

        args = init(verbose=0, log_interval=0)
        mml = MetaML(args, base=base)

        print('parameter scan : learning ')
        print(50*'-', 'Using SGD', 50*'-', sep='\n')

        for parameter in ['lr', 'momentum', 'batch_size', 'epochs'] :
            mml.parameter_scan(parameter)

        print(50*'-', 'Using ADAM', 50*'-', sep='\n')

        args = init(verbose=0, log_interval=0, do_adam=True)
        mml = MetaML(args, tag='adam')

        for parameter in ['lr', 'momentum', 'batch_size', 'epochs'] :
            mml.parameter_scan(parameter)
