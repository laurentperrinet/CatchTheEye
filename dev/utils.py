import matplotlib.pyplot as plt
import numpy as np
import torchvision
    
def show(img, gamma=.5, noise_level=.4, transpose=True):

    npimg = img.numpy()
    plt.figure()
    if transpose:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    else:
        plt.imshow(npimg)