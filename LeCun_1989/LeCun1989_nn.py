import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # initialization as described in the paper to my best ability, but it doesn't look right...
        winit = lambda fan_in, *shape: (torch.rand(*shape) - 0.5) * 2 * 2.4 / fan_in ** 0.5

        '''
        torch.nn.parameter.Parameter(data=None, requires_grad=True)
        '''

        self.H1w = nn.Parameter(winit(5*5*1, 12, 1, 5, 5))
        self.H1b = nn.Parameter(torch.zeros(12, 8, 8))

        self.H2w = nn.Parameter(winit(5*5*8, 12, 8, 5, 5))
        self.H2b = nn.Parameter(torch.zeros(12, 4, 4))

        self.H3w = nn.Parameter(winit(12*4*4, 12*4*4, 30))
        self.H3b = nn.Parameter(torch.zeros(30))

        self.outw = nn.Parameter(winit(30, 30, 10))
        self.outb = nn.Parameter(-1*torch.ones(10))

    def forward(self, x):
        '''
        torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        '''
        # x has shape (1, 1, 16, 16)
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0)  # pad by two using constant -1 for background
        x = F.conv2d(x, self.H1w, stride=2) + self.H1b
        x = torch.tanh(x)

        # x has shape (1, 12, 8, 8)
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0)  # pad by two using constant -1 for background
        slice1 = F.conv2d(x[:, 0:8], self.H2w[0:4], stride=2)  # first 4 filters look at first 8 input planes
        slice2 = F.conv2d(x[:, 4:12], self.H2w[4:8], stride=2)  # next 4 filters look at last 8 input planes
        slice3 = F.conv2d(torch.cat((x[:, 0:4], x[:, 8:12]), dim=1), self.H2w[8:12], stride=2)  # next 4 filters look at last 8 input planes
        x = torch.cat((slice1, slice2, slice3), dim=1) + self.H2b
        x = torch.tanh(x)
        '''
        torch.nn.functional.linear(input, weight, bias=None)
        '''
        x = x.flatten(start_dim=1)  # (1, 12*4*4)
        x = x @ self.H3w + self.H3b
        x = torch.tanh(x)

        x = x @ self.outw + self.outb
        x = torch.tanh(x)

        return x

    # -----------------------------------------------------------------------------

if __name__ == '__main__':
    '''This part of the code is completely from Karpathy's implementation'''

    parser = argparse.ArgumentParser(description="Train a 1989 LeCun ConvNet on digits")
    '''
     ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default]
     [, type][, choices][, required][, help][, metavar][, dest])¶
    '''
    parser.add_argument('--learning-rate', '-l', type=float, default=0.03, help="SGD learning rate")
    parser.add_argument('--output-dir', '-o', type=str, default='out/base', help="output directory for training logs")
    args = parser.parse_args()
    print(vars(args))

    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.use_deterministic_algorithms(True)

    # set up logging
    #Create Output directory

    os.makedirs(os.path.join(sys.path[0], args.output_dir), exist_ok=True)
    with open(os.path.join(sys.path[0],args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(args.output_dir)
    # init a model
    model = NeuralNetwork()
    # init data
    Xtr, Ytr = torch.load('train1989.pt')
    Xte, Yte = torch.load('test1989.pt')
    
    #init optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
    def eval_split(split):
        # eval the full train/test set, batched implementation for efficiency
        #model.eval() switches off some layers like Dropput layers, batchnormalization
        model.eval()
        X, Y = (Xtr, Ytr) if split == 'train' else (Xte, Yte)
        Yhat = model(X)
        loss = torch.mean((Y - Yhat)**2)
        err = torch.mean((Y.argmax(dim=1) != Yhat.argmax(dim=1)).float())
        print(f"eval: split {split:5s}. loss {loss.item():e}. error {err.item()*100:.2f}%. misses: {int(err.item()*Y.size(0))}")
        writer.add_scalar(f'error/{split}', err.item()*100, pass_num)
        writer.add_scalar(f'loss/{split}', loss.item(), pass_num)
    
    for pass_num in range(23):

        # perform one epoch of training
        model.train()
        
        for step_num in range(Xtr.size(0)):

            # fetch a single example into a batch of 1
            x, y = Xtr[[step_num]], Ytr[[step_num]]

            # forward the model and the loss
            yhat = model(x)
            loss = torch.mean((y - yhat)**2)

            # calculate the gradient and update the parameters
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # after epoch epoch evaluate the train and test error / metrics
        print(pass_num + 1)
        eval_split('train')
        eval_split('test')
    # save final model to file
    #print(model.state_dict())
    torch.save(model.state_dict(), os.path.join(sys.path[0],args.output_dir, 'model.pt'))
