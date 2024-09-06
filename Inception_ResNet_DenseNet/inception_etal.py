"""
This script implements popular Convolutional Neural Network (CNN) architectures namely, Inception (Szegedy et al. 2014),
ResNet (He et al. 2015), and DenseNet (Huang et al. 2016)
"""
# ---------------------------------------------------Required Imports------------------------------------------------- #
# --- Standard Libraries
import os
import json
import random
import numpy as np
from types import SimpleNamespace

# --- PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# --- Data
import urllib
from urllib.error import HTTPError
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

# --- Visualization
from PIL import Image
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from pytorch_lightning.loggers import TensorBoardLogger

# ---------------------------------------------------Reproducibility-------------------------------------------------- #
set_seed = 42

# --- For Python operations
random.seed(set_seed)
# --- For Numpy operations
np.random.seed(set_seed)
# --- PyTorch operations
torch.manual_seed(set_seed)

# ---  CUDA Operations
if torch.cuda.is_available():
    # Fixing seed for generating random number on the current GPU device
    torch.cuda.manual_seed(set_seed)
    # Fixing seed for generating random number on all (multiple) GPU devices
    torch.cuda.manual_seed_all(set_seed)

# --- Disable benchmarking for CUDA convolution operation
torch.backends.cudnn.benchmark = False
# ---  Avoid Non-deterministic Algorithms
torch.use_deterministic_algorithms(True)

# ---------------------------------------------------Dataset Paths---------------------------------------------------- #

dataset_path = './data'
# Path to the folder to store the trained models
checkpoint_path = "./inception/"
# Create the directory if not already created
os.makedirs(os.path.join(dataset_path), exist_ok=True)
os.makedirs(os.path.join(checkpoint_path), exist_ok=True)
# -----------------------------------------------Download & Transform the Data---------------------------------------- #
# --- Download the data once, estimate mean and standard deviation while composing transformation pipeline
train_dataset = CIFAR10(dataset_path, train=True, download=True)
mean = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
std = (train_dataset.data / 255.0).std(axis=(0, 1, 2))

# --- Compose Transformation Pipeline
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=mean, std=std)])
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.50),
                                      transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std)
                                      ])
# ---
train_set = CIFAR10(dataset_path, train=True, transform=train_transform, download=True)
validation_set = CIFAR10(dataset_path, train=True, transform=test_transform, download=True)

torch.manual_seed(set_seed)
train_dataset, _ = data.random_split(train_set, [45000, 5000])
torch.manual_seed(set_seed)
_, validation_dataset = data.random_split(validation_set, [45000, 5000])

test_dataset = CIFAR10(dataset_path, train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False, drop_last=True, num_workers=4)
test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

# -----------------------------------------------Visualize Augmented images------------------------------------------- #
sample_batch, sample_label = next(iter(train_loader))
# ---- Total number of images to be visualized
num_of_images = 4
# ---- Total number of images in this sample batch
total_num_images = train_set.data.shape[0]
indices = random.sample(range(total_num_images), num_of_images)
images = [train_set[index][0] for index in indices]  # These are Tensors, each image tensor of shape (3, 32, 32)
original_images = [Image.fromarray(train_set.data[index]) for index in indices]  # Create images from Tensors
original_images = [test_transform(img) for img in original_images]  # Transform back to Tensors and apply normalization
# Make a grid to display all images and their transformed versions
# Caution: Unlike nrow in Plotly, nrow here represents number of images to be displayed in a row
# Return type grid(Tensor)
image_grid = torchvision.utils.make_grid(torch.stack(images + original_images, dim=0), nrow=num_of_images,
                                         normalize=True, pad_value=0.5)
image_grid = image_grid.permute(1, 2, 0)

fig = px.imshow(image_grid, title='Augmentation examples on CIFAR10')
fig.update_layout(xaxis=dict(showgrid=False, showticklabels=False),
                  yaxis=dict(showgrid=False, showticklabels=False))
fig.show()
fig.write_html(os.path.join(checkpoint_path, 'data_augmentation_example.html'))

# -----------------------------------------------PyTorch Lightning---------------------------------------------------- #
# PyTorch Lightning - light PyTorch wrapper for automatic handling of Boilerplate code
# --- Seed everything
pl.seed_everything(42)

# ------ Model Creation Stage
# ---- Running multiple models through same Lightning module
model_dict = {}  # Initially empty, will store {'Inception': Inception(model_hparams)}


def create_model(model_name, model_hparams):
    """
    This function maps a model name to a model class
    :param model_name: Name of the model name to create the model class {Inception, DenseNet, ResNet}
    :param model_hparams: Model Hyperparameters
    :return:
    """
    if model_name in model_dict.keys():
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f"Unknown model name \ '{model_name}'. Available models include: {model_dict.keys()}"


# ------ General PyTorch Lightning instantiation, optimizer, training, validation, and testing logic - for all models
class CIFARModule(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Define the model architecture while instantiation
        :param model_name:
        :param model_hparams:
        :param optimizer_name:
        :param optimizer_hparams:
        """
        # Instantiate, inputs: model_name, model_hparams, optimizer_name, optimizer_hparams
        super().__init__()
        # --- Export the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # --- Create model
        self.model = create_model(model_name, model_hparams)
        # --- Create Loss module
        self.loss_module = nn.CrossEntropyLoss()
        # --- Example input of shape 32 x 32 with 3 channels, for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    # --- Define Forward hook for prediction/inference actions
    def forward(self, x):
        return self.model(x)

    # --- Optimizers go into configure_optimizers - currently only supporting Adam and SGD
    def configure_optimizers(self):
        # --- Values of hyperparameters stored in self.hparams
        if self.hparams.optimizer_name == 'Adam':
            # --- Weighted Adam is correct implementation of Adam
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == 'SGD':
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Optimizer object {self.hparams.optimizer_name} not supported at the moment"

        # --- Learning Rate Scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    # --- Training logic
    def training_step(self, train_batch, batch_index):
        images, labels = train_batch
        preds = self.model(images)
        loss = self.loss_module(preds, labels)
        # Expected shape (num_of_items in the batch, num_of_classes)
        # print(preds.shape)
        # Check why are we converting to float ?
        accuracy = (preds.argmax(dim=-1) == labels).float().mean()

        # --- Tensorboard Logging
        # Adding on_epoch=True would accumulate
        self.log('Training_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('Training_Loss', loss, on_epoch=True)

        # loss return type tensor to call loss.backward() on
        return loss

    # # --- Train Loader
    # def train_dataloader(self):
    #     return DataLoader()
    # --- Validation Step
    def validation_step(self, val_batch, batch_index):
        images, labels = val_batch
        preds = self.model(images)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # Calling self.log from validation step will automatically accumulate and log at the end of the epoch
        self.log('Validation_accuracy', acc)

    def test_step(self, test_batch, batch_index):
        images, labels = test_batch
        preds = self.model(images)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # Calling self.log from validation step will automatically accumulate and log at the end of the epoch
        self.log('Test_accuracy', acc)


# ------------------------------------------------Tensorboard Logger-------------------------------------------------- #
# Create a TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="my_model")


# ----- Trainer module of PyTorch Lightning is responsible to execute the training steps defined in the lightning module
#       and complete the framework. Basic working:
#           1. trainer.fit: Take as input lightning module, a training dataset, and an (optional) validation dataset.
#                           This function trains the given module on the training dataset with occasional validation
#                           (defaults once per epoch, can be changed)
#           2. trainer.test: Takes as input a model and test dataset and returns test metric on the dataset

def train_model(model_name, save_name=None, **kwargs):
    """
    :param model_name: str, Name of the model to run, this will be passed through a dictionary to run Inception, ResNet, or DenseNet
    :param save_name: (optional) str, if specified will be used to save the given model
    :param kwargs: additional input keyword arguments
    :return: results (dict) Accuracy results on Validation and Test sets
    """
    # --- Check if save_name has a value
    if save_name is None:
        save_name = model_name

    # --- Define trainer object
    trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, save_name),  # Defaults to os.getcwd()
                         logger=logger,
                         accelerator='cpu',  # currently gpu not available
                         devices=1,
                         max_epochs=10,
                         callbacks=[ModelCheckpoint(monitor='Validation_accuracy', mode='max', save_weights_only=True),
                                    LearningRateMonitor(logging_interval='epoch')],
                         enable_progress_bar=True
                         )

    # --- Plot Computation Graph on Tensorboard Logger
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    # --- Check if pre-trained model(s) exist. If yes, skip training or else trainer.fit()
    pretrained_filename = os.path.join(checkpoint_path, save_name + '.ckpt')
    print(f"Pretrained: {pretrained_filename}")
    if os.path.isfile(pretrained_filename):
        print(f"Pre-trained model file already exists. Loading")
        model = CIFARModule.load_from_checkpoint(pretrained_filename)
    else:
        # --- Seed everything for reproducibility
        pl.seed_everything(42)
        model = CIFARModule(model_name=model_name, **kwargs)

        # ---- trainer.fit()
        trainer.fit(model, train_loader, val_loader)

        # ---- Load best model after training, best defined as maximum validation_accuracy
        model = CIFARModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # ---- Test the best model
    test_results = trainer.test(model, test_loader, verbose=False)
    val_results = trainer.test(model, val_loader, verbose=False)

    results = {'test': test_results[0]["Test_accuracy"], 'validation': val_results[0]['Test_accuracy']}
    print(f"printing results: {results}")
    return model, results


# -----------------------------------------------Inception Block------------------------------------------------------ #
class Inception_Block(nn.Module):

    def __init__(self, c_in, c_red: dict, c_out: dict, act_function):
        """
        :param c_in: int, Number of input channels from the previous layers
        :param c_red: dict, This is for reducing number of channels before 3x3 and 5x5 operations. Keys = ['3x3', '5x5']
        :param c_out: dict, This is before concatenation layer. Keys = ['1x1', '3x3', '5x5', 'max']
        :param act_function: constructor, for applying the appropriate activation function
        """
        super().__init__()

        # ---- 1x1 convolution
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out['1x1'], kernel_size=1),
            nn.BatchNorm2d(c_out['1x1']),
            act_function()
        )

        # ---- 3x3 convolution
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_red['3x3'], kernel_size=1),
            nn.BatchNorm2d(c_red['3x3']),
            act_function(),
            nn.Conv2d(in_channels=c_red['3x3'], out_channels=c_out['3x3'], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out['3x3']),
            act_function()
        )

        # ---- 5x5 convolution
        self.conv_5x5 = nn.Sequential(nn.Conv2d(in_channels=c_in, out_channels=c_red['5x5'], kernel_size=1),
                                      nn.BatchNorm2d(c_red['5x5']),
                                      act_function(),
                                      nn.Conv2d(in_channels=c_red['5x5'], out_channels=c_out['5x5'], kernel_size=5,
                                                padding=2),
                                      nn.BatchNorm2d(c_out['5x5']),
                                      act_function()

                                      )

        # ---- max pooling
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels=c_in, out_channels=c_out['max'], kernel_size=1),
            nn.BatchNorm2d(c_out['max']),
            act_function()
        )

    def forward(self, x):
        x_conv1x1 = self.conv_1x1(x)
        x_conv3x3 = self.conv_3x3(x)
        x_conv5x5 = self.conv_5x5(x)
        x_maxpool = self.max_pool(x)
        x = torch.cat([x_conv1x1, x_conv3x3, x_conv5x5, x_maxpool], dim=1)
        return x


# -----------------------------------------GoogleNet: Szegedy etal. 2014---------------------------------------------- #
"""
This implementation is a reduced version as the original architecture Szegedy et al. 2014 was applied to 224 x 224 images.
However, CIFAR images are 32 x 32 so, implementing a reduced version here
"""
# Activation function dict
activation_func = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'leakyrelu': nn.LeakyReLU,
    'gelu': nn.GELU
}


class GoogleNet(nn.Module):
    def __init__(self, num_classes=10, act_function_name='relu', **kwargs):
        super().__init__()
        self.hparams = SimpleNamespace(num_classes=num_classes,
                                       act_func_name=act_function_name,
                                       act_function=activation_func[act_function_name])
        self._create_network()
        self._init_params()

    def _create_network(self):
        # -- First convolution Layer, followed by Batch Normalization and Activation function
        # ---- Training.shape = (Batch_size, 3, 32, 32)
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.hparams.act_function()
        )

        # -- Inception Blocks
        self.inception_blocks = nn.Sequential(
            # Two Inception Blocks
            # ---- Training.shape = (Batch_size, 64, 32, 32)
            Inception_Block(64, c_red={'3x3': 32, '5x5': 16}, c_out={'1x1': 16, '3x3': 32, '5x5': 8, 'max': 8},
                            act_function=self.hparams.act_function),
            # ---- Training.shape = (Batch_size, 64, 32, 32)
            Inception_Block(64, c_red={'3x3': 32, '5x5': 16}, c_out={'1x1': 24, '3x3': 48, '5x5': 12, 'max': 12},
                            act_function=self.hparams.act_function),
            # ---- Training.shape = (Batch_size, 96, 32, 32)
            # Reducing Layer
            nn.MaxPool2d(3, stride=2, padding=1),  # Reduces 32x32 to 16x16;
            # ---- Training.shape = (Batch_size, 96, 16, 16)
            # Four Inception Blocks
            Inception_Block(96, c_red={'3x3': 32, '5x5': 16}, c_out={'1x1': 24, '3x3': 48, '5x5': 12, 'max': 12},
                            act_function=self.hparams.act_function),
            Inception_Block(96, c_red={'3x3': 32, '5x5': 16}, c_out={'1x1': 16, '3x3': 48, '5x5': 16, 'max': 16},
                            act_function=self.hparams.act_function),
            Inception_Block(96, c_red={'3x3': 32, '5x5': 16}, c_out={'1x1': 16, '3x3': 48, '5x5': 16, 'max': 16},
                            act_function=self.hparams.act_function),
            Inception_Block(96, c_red={'3x3': 32, '5x5': 16}, c_out={'1x1': 32, '3x3': 48, '5x5': 24, 'max': 24},
                            act_function=self.hparams.act_function),
            # ---- Training.shape = (Batch_size, 128, 16, 16)
            # Reducing Layer
            nn.MaxPool2d(3, stride=2, padding=1),  # Reduces 16x16 to 8x8;
            # ---- Training.shape = (Batch_size, 128, 8, 8)
            # Two Inception Blocks
            Inception_Block(128, c_red={'3x3': 48, '5x5': 16}, c_out={'1x1': 32, '3x3': 64, '5x5': 16, 'max': 16},
                            act_function=self.hparams.act_function),
            Inception_Block(128, c_red={'3x3': 48, '5x5': 16}, c_out={'1x1': 32, '3x3': 64, '5x5': 16, 'max': 16},
                            act_function=self.hparams.act_function)
            # ---- Training.shape = (Batch_size, 128, 8, 8)

        )

        # -- Flattening layers
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            # ---- Training.shape = (Batch_size, 128, 1, 1)
            nn.Flatten(),
            # ---- Training.shape = (Batch_size, 128)
            nn.Linear(128, self.hparams.num_classes)
        )

    # For ReLU activation using Kaiming et al. 2014 initialization
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_func_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x


model_dict["GoogleNet"] = GoogleNet
# googlenet_model, googlenet_results = train_model(model_name="GoogleNet",
#                                                  model_hparams={"num_classes": 10,
#                                                                 "act_function_name": "relu"},
#                                                  optimizer_name="Adam",
#                                                  optimizer_hparams={"lr": 1e-3,
#                                                                     "weight_decay": 1e-4})
# print(f"GoogleNet Results: {googlenet_results}")


# -----------------------------------------ResNet: He et al. 2015----------------------------------------------------- #
# ---- ResNet (Identity Mappings in Deep Residual Networks, He et al. 2016), Implementation of Original Residual Unit
# ---- Reference Figure1a in He et al. 2016

class ResNetBlock(nn.Module):
    def __init__(self, c_in, act_func, subsample=False, c_out=-1):
        """
        :param c_in: int, Number of channels in the input
        :param act_func: object, Activation class constructor
        :param subsample: bool, if True reduce the size of input by half using stride =2
        :param c_out: int, Number of channels in the output
        """
        super().__init__()

        if not subsample:
            c_out = c_in

        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2 if subsample else 1, bias=False),
            nn.BatchNorm2d(c_out),
            act_func(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out)

        )
        # 1x1 convolution to reduce the size of output
        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = act_func()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)

        output = z + x
        output = self.act_fn(output)
        return output


# ---- ResNet (Identity Mappings in Deep Residual Networks, He et al. 2016), Implementation of Proposed Residual Unit
# ---- Reference Figure1b in He et al. 2016
class PreActResNetBlock(nn.Module):
    def __init__(self, c_in, act_func, subsample=False, c_out=-1):
        """
        :param c_in: int, Number of channels in the input
        :param act_func: object, Activation class constructor
        :param subsample: bool, if True reduce the size of input by half using stride =2
        :param c_out: int, Number of channels in the output
        """
        super().__init__()

        if not subsample:
            c_out = c_in

        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_func(),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2 if subsample else 1, bias=False),
            nn.BatchNorm2d(c_out),
            act_func(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),

        )
        # 1x1 convolution to reduce the size of output
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_func(),
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False)
        ) if subsample else None
        self.act_fn = act_func()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)

        output = z + x
        return output


# ---- Dictionary to choose and call ResNet block from
resnet_names_dict = {'ResNetBlock': ResNetBlock,
                     'PreActResNetBlock': PreActResNetBlock}

# ---- ResNet Architecture; num_blocks = [3, 3, 3], first block of each group uses downsampling except first one
# ---- Activation function used ReLU, as specified in the paper
# ---- Optimization function = SGD, Adam has been shown to perform worse

class ResNet(nn.Module):

    def __init__(self, num_classes=10, num_blocks=[3, 3, 3], c_hidden=[16, 32, 64], act_fn_name='relu', block_name='ResNetBlock', **kwargs):
        super().__init__()
        assert block_name in resnet_names_dict, f"Block Name: {block_name} is not defined"
        assert act_fn_name in activation_func, f"Activation Function named {act_fn_name} is not defined currently"
        self.hparams = SimpleNamespace(num_classes=num_classes,
                                       c_hidden=c_hidden,
                                       num_blocks=num_blocks,
                                       act_func_name=act_fn_name,
                                       act_function=activation_func[act_fn_name],
                                       block_class=resnet_names_dict[block_name])
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden

        # ---- Initial convolution on the original image to increase
        #      the number of channels from 3 to specified in c_hidden[0]

        if self.hparams.block_class == PreActResNetBlock:
            self.input_net = nn.Sequential(
                nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False)
            )
        else:
            self.input_net = nn.Sequential(
                nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_hidden[0]),
                self.hparams.act_function()
            )

        # ---- ResNet Blocks
        blocks = []
        for block_index, block_count in enumerate(self.hparams.num_blocks):
            # ---- If not first block then downsample
            for count in range(block_count):
                # --- Only for first count for each block except for the first block
                subsample = (count == 0 and block_index > 0)
                blocks.append(self.hparams.block_class(c_in=c_hidden[block_index if not subsample else (block_index-1)],
                                                       act_func=self.hparams.act_function,
                                                       subsample=subsample,
                                                       c_out=c_hidden[block_index]))

        self.blocks = nn.Sequential(*blocks)

        # ---- Output Block
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.hparams.num_classes)
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.hparams.act_func_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x


model_dict["ResNet"] = ResNet
resnet_model, resnet_results = train_model(model_name="ResNet",
                                           model_hparams={"num_classes": 10,
                                                          "c_hidden": [16, 32, 64],
                                                          "num_blocks": [3, 3, 3],
                                                          "act_function_name": "relu"},
                                           optimizer_name="SGD",
                                           optimizer_hparams={"lr": 0.1,
                                                              "momentum": 0.9,
                                                              "weight_decay": 1e-4})
print(f"ResNet Results: {resnet_results}")

# -----------------------------------------DenseNet: Huang et al. 2016------------------------------------------------ #
# ---- DenseNet (Combine ideas from ResNet and Inception)
#           ResNet: Adding input such that the in-between layers only modelled the residuals
#           Inception: Using both 3x3 and 5x5 convolution, concatenating the output from different filter sizes
# ---- Three important components:
#           1. Dense Layer:
#               a. Architecture:
#                   BatchNorm->ReLU->1x1 conv, channel size = bottleneck_size*growth_rate->BatchNorm->ReLU->3x3 conv.
#               b. Forward: Output = concatenate(input, self.DenseLayer(input))
#           2. Dense Block:
#               a. Architecture:
#                   Number of Dense Layers,
#               b. Forward: self.DenseBlock(input)
#           3. Transition Layer:
#               a. Architecture: BatchNorm-> ReLU->1x1 Conv->Average Pooling Layer

# ---- DenseLayer
class DenseLayer(nn.Module):
    def __init__(self, c_in, bn_size, growth_rate, act_func):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_func(),
            nn.Conv2d(c_in, bn_size*growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size*growth_rate),
            act_func(),
            nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        output = self.net(x)
        # We end up increasing the channel size here
        # Every output would carry forward the input vector as well
        # dim=0: Batch_size, dim =1: Num. of channels, dim =2:
        output = torch.cat([output, x], dim=1)
        return output

# ---- DenseBlock

class DenseBlock(nn.Module):

    def __init__(self, c_in, num_layers, bn_size, growth_rate, act_func):
        super().__init__()
        layers = []
        for layer_index in range(num_layers):
            layers.append(DenseLayer(c_in=c_in+layer_index*growth_rate,
                                     bn_size=bn_size,
                                     growth_rate=growth_rate,
                                     act_func=act_func
                                     )
                          )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# ---- Transition Layer

class TransitionLayer(nn.Module):
    def __init__(self, c_in, c_out, act_func):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_func(),
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

# ---- Defining the complete network
# ---- Combined Architecture:
#       a. Input Layer: Up scale the channels in the input image from 3 to 32
#       b. Blocks: Several DenseBlocks and Transition layers combined
#       c. Output Layer: BatchNorm->ReLU->AdaptiveAveragePooling->Flatten->Linear(c_hidden, num_classes)

class DenseNet(nn.Module):
    def __init__(self, num_classes=10, num_layers=[6, 6, 6, 6, 6], bn_size=2, growth_rate=16, act_fn_name='relu', **kwargs):
        super().__init__()
        # ---- Write the hyperparameters into self.hparams
        self.hparams = SimpleNamespace(num_classes=num_classes,
                                       num_layers=num_layers,
                                       bn_size=bn_size,
                                       growth_rate=growth_rate,
                                       act_fn_name=act_fn_name,
                                       act_function=activation_func[act_fn_name]
                                       )
        self._create_network()
        self._init_params()

        # ---- Create Network

    def _create_network(self):
        c_hidden = self.hparams.growth_rate * self.hparams.bn_size

        # ---- Input Layer
        self.input_net = nn.Sequential(
            nn.Conv2d(3, c_hidden, kernel_size=3, padding=1)
            )

        # ---- DenseBlocks
        blocks = []
        for block_index, num_layers in enumerate(self.hparams.num_layers):
            blocks.append(DenseBlock(c_in=c_hidden,
                                     num_layers=num_layers,
                                     bn_size=self.hparams.bn_size,
                                     growth_rate=self.hparams.growth_rate,
                                     act_func=self.hparams.act_function)
                          )
            c_hidden = c_hidden + num_layers * self.hparams.growth_rate
            # Except for the last DenseBlock apply transition layer after each Dense Block
            if not block_index >= len(self.hparams.num_layers)-1:
                blocks.append(TransitionLayer(c_in=c_hidden,
                                              c_out=c_hidden//2,
                                              act_func=self.hparams.act_function)
                                  )
                c_hidden = c_hidden//2
        self.blocks = nn.Sequential(*blocks)
        # ---- Output Layer
        self.output_net = nn.Sequential(
                nn.BatchNorm2d(c_hidden),
                self.hparams.act_function(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(c_hidden, self.hparams.num_classes)
            )
        # ---- Initialize Parameters of the model - Kaiming for ReLU
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    # ---- Define Forward pass
    def forward(self,x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x


model_dict["DenseNet"] = DenseNet
densenet_model, densenet_results = train_model(model_name="DenseNet",
                                               model_hparams={"num_classes": 10,
                                                              "num_layers": [6,6,6,6],
                                                              "bn_size": 2,
                                                              "growth_rate": 16,
                                                              "act_fn_name": "relu"},
                                               optimizer_name="Adam",
                                               optimizer_hparams={"lr": 1e-3,
                                                                  "weight_decay": 1e-4})
