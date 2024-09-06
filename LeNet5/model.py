'''
This is an implementation of LeCun, Bottou, Bengio, and Haffner 1998, popularly known as LeNet5
Reference: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
'''
# Required imports
import os
import sys
import argparse

import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from PIL import Image
import plotly.express as px

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For reproducibility
torch.manual_seed(881)
np.random.seed(881)

# Build the model
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        # Fully-connected layers
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)
    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0)
        x = F.max_pool2d(torch.tanh(self.conv1(x)), (2, 2))
        x = F.max_pool2d(torch.tanh(self.conv2(x)), (2, 2))
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a 1989 LeCun ConvNet on digits")
    parser.add_argument('--learning-rate', '-l', type=float, default=0.03, help="SGD learning rate")
    parser.add_argument('--output-dir', '-o', type=str, default='out/base', help="output directory for training logs")
    parser.add_argument('--batch-size', '-bs', type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument('--epochs', '-e', type=int, default=5, help="Number of Epochs for training data")
    args = parser.parse_args()
    print(vars(args))
    os.makedirs(os.path.join(sys.path[0], args.output_dir), exist_ok=True)
    # Experiment settings
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    # Pipeline to convert an image to a tensor
    # Consider applying transforms.Resize(20) later
    transform_pipeline = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0), (1))])

    training_data = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform_pipeline)

    testing_data = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform_pipeline)

    # Visualize a random training example -for good measure
    sample_idx = 97
    transform_img = transforms.ToPILImage()
    fig = px.imshow(transform_img(training_data[sample_idx][0]))
    # Alternatively can simply use fig.show() to open html file on local port
    fig.write_html(os.path.join(sys.path[0], args.output_dir ,f"Training_example{sample_idx}.html"))

    # Apply dataloader
    train_loader = DataLoader(dataset=training_data,
                              batch_size=args.batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=testing_data,
                              batch_size=args.batch_size,
                              shuffle=True)
    # Model
    model = LeNet5()
    print(model)
    # init optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    # Loss function
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        # for displaying later
        loss_val = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            #zero-out the gradient
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_val += loss.item()
            if i % 100 == 99:
                print(f"[{epoch+1}, {i+1 :5}] loss: {loss_val/100 :.3f}")
                loss_val = 0.0

    print('----Completed Training!----')
    correct = 0
    total = 0
    #Testing
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the trained model is {100*(correct/total)}")













