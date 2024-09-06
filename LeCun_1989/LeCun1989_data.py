'''
This exercise is an implementation of LeCun et al. 1989 paper titled
'Backpropagation Applied to Handwritten Zip Code Recognition' in PyTorch, inspired by
Andrej Karapathy's blog https://karpathy.github.io/2022/03/14/lecun1989/

According to the paper the training and testing set consisted of 7291 and 2007 images, respectively.
'''

#Import Required Packages
import os
import sys
import torch
import numpy as np
from torchvision import datasets #To extract the MNIST data
from torchvision import transforms #Transform input images from the dataset to apt. Tensors

torch.manual_seed(1337)
np.random.seed(1337)

for split in {'train', 'test'}:

    data = datasets.MNIST('./data', train=(split=='train'), download=True)
    #The standard dataset consists of 60K and 10K images, respectively, in the training and the test set.

    #Following the original split given in the paper
    n = 7291 if split == 'train' else 2007

    #Indices of image to be included in training and test sets
    img_index = np.random.permutation(len(data))[:n]

    '''
    torch.full(size, fill_value, dtype, requires_grad)
    :param size (int...) – a list, tuple, or torch.Size of integers defining the shape of the output tensor
    :param  fill_value (Scalar) – the value to fill the output tensor with.
    
    :keyword dtype (torch.dtype, optional) – the desired data type of returned tensor
    :keyword requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.
    '''
    # Create empty tensors to store X and Y values, size of an image in the dataset now is 28 * 28 while the paper originally used 16 *16
    # The Y-vector should contain values from -1 to 1 (acc. to Andrej Karpathy),
    # where the vector position for correct digit would get a 1.0 and the rest would get -1.0
    X = torch.full((n, 1, 16, 16), 0.0, dtype=torch.float32)
    Y = torch.full((n, 10), -1.0, dtype=torch.float32)

    '''
       Pipeline to 
       1. Resize the PIL image from 28 * 28 to 16 *16
       2. Convert the Resized PIL image to a Tensor
    '''
    transform_pipeline = transforms.Compose([transforms.Resize((16)), transforms.PILToTensor()])

    # Iterate through the list of randomly selected image indices and store them in X and Y tensors
    for i, index in enumerate(img_index):
        a, b = data[int(index)]
        X[i] = transform_pipeline(a)/127.5 -1
        Y[i, b] = 1.0

    '''
    torch.save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True)
    :param  obj (object) – saved object
    :param f (Union[str, PathLike, BinaryIO, IO[bytes]]) - a file-like object (has to implement write and flush)
                                                          or a string or os.PathLike object containing a file name
    '''
    torch.save((X, Y), os.path.join(sys.path[0], f"{split}1989.pt"))
