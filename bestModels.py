#----------------------------------------------------------------------------------------------------------------
# Import libraries
"""Libraries to managae directories"""
import os
import gdown
import traceback
import sys
import importlib
from pathlib import Path
from zipfile import ZipFile

"""Libraries to manage data"""
import math
import random
import argparse
import time
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Dict, Optional, Callable

"""Libraries to manage datasets"""
import json
import tarfile
import subprocess
import pickle
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

"""Libraries to manage audios"""
import boto3
import soundfile as sf
import click
import yaml
import glob
import tensorboard
import parallel_wavegan
import pydub

import transformers
from transformers import AutoModel

import munch
from munch import Munch

"""Scipy"""
import scipy
from scipy.spatial.distance import cdist
from scipy import signal
from scipy.io import wavfile

"""Torchaudio"""
import torchaudio
import torchaudio.transforms as T
from torchaudio.functional import resample

"""Pytorch"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import torch.optim as optim

"""Librosa"""
import librosa
import librosa.display
from librosa.display import waveshow, specshow
import librosa.feature
import librosa.feature.inverse
import librosa.effects

"""IPython"""
import IPython
from IPython import display
from IPython.display import Audio, display

"""Functions to prepare the datasets"""
#----------------------------------------------------------------------------------------------------------------
# Padding of the MFCCs
def pad_sequence(dataset):
    """Pad all MFCCs in dataset['MFCCs'] to the same length"""

    mfcc_list = []
    for mfcc in dataset['MFCCs']:
        if isinstance(mfcc, np.ndarray):
            mfcc_list.append(torch.from_numpy(mfcc).float())
        else:
            mfcc_list.append(mfcc.float())

    # Apply padding
    mfcc_list = [item.t() for item in mfcc_list]
    mfcc_list = torch.nn.utils.rnn.pad_sequence(mfcc_list, batch_first=True, padding_value=0.)
    mfcc_list = mfcc_list.permute(0, 2, 1)

    dataset['MFCCs'] = mfcc_list

    return dataset

#----------------------------------------------------------------------------------------------------------------
# Splitting of the data
class SubsetSC():
    def __init__(
            self,
            dataset: dict,
            official_split_root: str = None
        ):

        self.dataset = dataset

        # Load official split files:
        val_list_path = os.path.join(official_split_root, "validation_list.txt")
        test_list_path = os.path.join(official_split_root, "testing_list.txt")

        with open(val_list_path, 'r') as f:
            val_files = set(line.strip() for line in f)

        with open(test_list_path, 'r') as f:
            test_files = set(line.strip() for line in f)

        # Match paths to splits
        self.train_idx = []
        self.val_idx = []
        self.test_idx = []

        for i, path in enumerate(dataset['files']):
            # Extract relative path
            normalized_path = path.replace('\\', '/')
            parts = normalized_path.split('/')[-2:]
            rel_path = '/'.join(parts)

            if rel_path in test_files:
                self.test_idx.append(i)
            elif rel_path in val_files:
                self.val_idx.append(i)
            else:
                self.train_idx.append(i)

        self.train_idx = np.array(self.train_idx)
        self.val_idx = np.array(self.val_idx)
        self.test_idx = np.array(self.test_idx)

    def get_split(self, split: str = "train"):

        if split == 'train':
            idx = self.train_idx
        elif split == 'val':
            idx = self.val_idx
        elif split == 'test':
            idx = self.test_idx
        # Constraint on split name
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        # Create new dictionary with selected indices
        return {
            'MFCCs': [self.dataset['MFCCs'][i] for i in idx],
            'labels': [self.dataset['labels'][i] for i in idx],
            'files': [self.dataset['files'][i] for i in idx],
            'mapping': self.dataset['mapping']
        }
    
#----------------------------------------------------------------------------------------------------------------
# Preparing the data to pass in the dataloader
class prepareDataset(Dataset):
    def __init__(self, data_dict):
        self.mfccs = data_dict['MFCCs']
        self.labels = data_dict['labels']
        self.files = data_dict['files']
        self.mapping = data_dict['mapping']

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, idx):
        """
        Returns:
            mfcc: The MFCC features (as tensor)
            label: The corresponding label (as tensor)
        """
        mfcc = torch.FloatTensor(self.mfccs[idx])
        label = torch.LongTensor([self.labels[idx]])[0]  # Convert to scalar tensor

        return mfcc, label

"""Created models"""
#----------------------------------------------------------------------------------------------------------------
# SIMPLE CNN (from the article "Adversial Example detection by classification for deep speech recognition")
class simpleCNN(nn.Module):
    def __init__(
            self,
            input_shape,
            n_output = 10    # Number of possible labels
        ):
        super(simpleCNN, self).__init__()

        # Store input shape for reference
        self.input_shape = input_shape

        # 1st conv layer -- activation: ReLU
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 2)) # L2 kernel regularizer during the training process
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3))

        # 2nd conv layer -- activation: ReLU
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(2, 2)) # L2 kernel regularizer during the training process
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # 3rd conv layer -- activation: SeLU
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(2, 2)) # L2 kernel regularizer during the training process
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.4)

        # Fully connected layers -- activation: ReLU and softmax
        self.flattened_size = self._get_flattened_size(input_shape)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_output)

    def _get_flattened_size(self, input_shape):
        """Calculate the size after conv and pooling layers"""
        with torch.no_grad():

            # Constraint on input shape
            if len(input_shape) == 2:
                h, w = input_shape # Ensure input_shape is (height, width)
            else:
                raise ValueError(f"Expected 2D input_shape, got shape: {input_shape}")

            dummy_input = torch.zeros(1, 1, h, w)

            x = self.conv1(dummy_input)
            x = F.relu(self.bn1(x))
            x = self.pool1(x)

            x = self.conv2(x)
            x = F.relu(self.bn2(x))
            x = self.pool2(x)

            x = self.conv3(x)
            x = F.selu(self.bn3(x))
            x = self.pool3(x)

            # Get flatten size
            flattened_size = x.view(1, -1).size(1)

            return flattened_size

    def forward(self, x):

        # Ensure input has channel dimension: (batch, channels, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Conv block 1 (with ReLU)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        # Conv block 2 (with ReLU)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        # Conv block 3 (with SeLU)
        x = self.conv3(x)
        x = F.selu(self.bn3(x))
        x = self.pool3(x)
        x = self.dropout1(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

#----------------------------------------------------------------------------------------------------------------
# COMPLEX CNN (from the article "Trojaning attack on Neural Networks")
class complexCNN(nn.Module):
    def __init__(
            self,
            input_shape,
            n_output = 10    # Number of possible labels
        ):
        super(complexCNN, self).__init__()

        # 1st conv layer -- activation: None
        self.conv1 = nn.Conv2d(1, 96, kernel_size=(3, 3), padding=1) # L2 kernel regularizer during the training process
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # 2nd conv layer -- activation: None
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(3, 3), padding=1) # L2 kernel regularizer during the training process
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # 3rd conv layer -- activation: ReLU
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1) # L2 kernel regularizer during the training process

        # 4th conv layer -- activation: ReLU
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1) # L2 kernel regularizer during the training process

        # 5th conv layer -- activation: ReLU
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1) # L2 kernel regularizer during the training process
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        # Dense layers -- activation: ReLU
        self.flattened_size = self._get_flattened_size(input_shape)
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, n_output)

    def _get_flattened_size(self, input_shape):
        """Calculate the size after conv and pooling layers"""
        with torch.no_grad():

            # Constraint on input shape
            if len(input_shape) == 2:
                h, w = input_shape # Ensure input_shape is (height, width)
            else:
                raise ValueError(f"Expected 2D input_shape, got shape: {input_shape}")

            dummy_input = torch.zeros(1, 1, h, w)

            x = self.conv1(dummy_input)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.pool2(x)

            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            x = self.pool3(x)

            # Get flatten size
            flattened_size = x.view(1, -1).size(1)

            return flattened_size

    def forward(self, x):

        # Conv block 1 (no activation)
        x = self.conv1(x)
        x = self.pool1(x)

        # Conv block 2 (no activation)
        x = self.conv2(x)
        x = self.pool2(x)

        # Conv blocks 3-5 (with ReLU)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

#----------------------------------------------------------------------------------------------------------------
# SPEECH-SPECIFIC LSTM (from the article "A neural attention model for speech command recognition")
class speechSpecificLSTM(nn.Module):
    def __init__(
            self,
            n_output=10, 
            lstm_hidden=64, 
            attention_dim=128
        ):
        super(speechSpecificLSTM, self).__init__()

        # 1st conv layer -- activation: ReLU
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 1)) # (5,1) kernels for temporal processing
        self.bn1 = nn.BatchNorm2d(10)

        # 2nd conv layer -- activation: ReLU
        self.conv2 = nn.Conv2d(10, 1, kernel_size=(5, 1)) # (5,1) kernels for temporal processing
        self.bn2 = nn.BatchNorm2d(1)

        # Bidirectional LSTM layers: capture 2-way long term dependencies in the audio file
        self.lstm1 = nn.LSTM(
            input_size = 40,              # Input: 40-mel bands
            hidden_size = lstm_hidden,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )

        self.lstm2 = nn.LSTM(
            input_size = lstm_hidden * 2,  # *2 because bidirectional
            hidden_size = lstm_hidden,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )

        # Attention mechanism: query projection from last hidden state
        self.query_proj = nn.Linear(lstm_hidden * 2, attention_dim)

        # Dense layers after attention
        self.fc1 = nn.Linear(lstm_hidden * 2, 64) # Uses ReLU activation
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32) # Uses ReLU activation
        self.fc3 = nn.Linear(32, n_output) # Uses Softmax activation

    def attention(self, query, keys_values):
        """Dot-product attention mechanism"""

        # Expand query
        query_expanded = query.unsqueeze(1)

        # Compute attention scores using the dot product (matrix-matrix product)
        att_scores = torch.bmm(query_expanded, keys_values.transpose(1, 2))

        # Remove middle dimension
        att_scores = att_scores.squeeze(1)

        # Apply softmax
        att_weights = F.softmax(att_scores, dim=1)
        att_weights_expanded = att_weights.unsqueeze(1)

        # Compute the most relevant part of the audio
        context = torch.bmm(att_weights_expanded, keys_values)
        context = context.squeeze(1)

        return context, att_weights

    def forward(self, x):

        # Prepare input
        x = x.transpose(2, 3)

        # Conv block 1
        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        # Conv block 2
        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        # Squeeze channel dimension
        x = x.squeeze(1)

        # First Bidirectional LSTM
        x, _ = self.lstm1(x)
        # Second Bidirectional LSTM
        x, _ = self.lstm2(x)

        # Get last timestep for query
        x_last = x[:, -1, :]

        # Project to query space
        query = self.query_proj(x_last)
        # Apply attention mechanism
        context, att_weights = self.attention(query, x)

        # Dense layers
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

#----------------------------------------------------------------------------------------------------------------
# ORIGINAL LSTM (from the article "Long Short-Term memory")
class originalLSTM(nn.Module):
    def __init__(
            self,
            hidden_size=64, 
            num_layers=2, 
            n_output=10, 
            dropout=0.4
        ):
        super(originalLSTM, self).__init__()

        # Set parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Preprocessing layer
        self.bn = nn.BatchNorm2d(1)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size = 40,            # 40-mel bands
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0
        )

        # Output dense layer
        self.fc = nn.Linear(hidden_size, n_output)

    def forward(self, x):

        # Preprocessing
        x = F.relu(self.bn(x))

        # Get input dimensions
        batch_size, channels, height, width = x.shape

        # Reshape for LSTM
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(batch_size, width, height * channels)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(last_output)

        return F.log_softmax(out, dim=1)

#----------------------------------------------------------------------------------------------------------------
# ResNet-18 (from the article "Deep residual learning for image recognition")
class BasicResidualBlock(nn.Module):
    """
    Basic residual block for ResNet-18 with the following structure:
    - Conv 3x3 -> BN -> ReLU
    - Conv 3x3 -> BN
    - Add residual connection
    - ReLU
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Add residual connection
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet18(nn.Module):
    def __init__(
            self, 
            n_output=10
        ):
        super(ResNet18, self).__init__()

        self.inplanes = 64

        # Initial convolution block
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 64,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers (ResNet-18 has [2, 2, 2, 2] blocks)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate the feature size after all layers
        self.fc = nn.Linear(512, n_output)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, planes, blocks, stride=1):
        """Create a residual layer with multiple BasicBlocks"""

        downsample = None

        # Constraint on when to downsamples
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        # First block (may include downsampling)
        layers.append(BasicResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicResidualBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        # Initial conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

"""Definition of the class for the training of the models"""
class modelTrainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            loss: nn.Module,
            optimizer: optim.Optimizer,
            num_epochs: int,
            save_dir: str = "./checkpoints",
            experiment_name: str = "experiment",
            scheduler: Optional[object] = None,
            patience: int = 20,
            l2_reg: bool = False,
            lambda_reg: float = 0.001,
            verbose: bool = True
        ):
        """Set up training process"""

        # Automatically detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Paramters for the training of the models
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss = loss
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.patience = patience
        self.l2_reg = l2_reg
        self.lambda_reg = lambda_reg
        self.verbose = verbose

        # Setup save directory
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        if self.verbose:
            print(f"Training on device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def compute_l2_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss for convolutional layers"""
        l2_loss = torch.tensor(0., device=self.device)

        for module in self.model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Add L2 norm of the weights
                l2_loss += torch.norm(module.weight, p=2)

        return self.lambda_reg * l2_loss

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch using the train dataloader"""
        self.model.train()
        running_loss = 0.0
        running_classification_loss = 0.0
        running_reg_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} [Train]",
                    disable=not self.verbose)

        for batch_idx, (data, target) in enumerate(pbar):

            # Ensure correct input shape (batch, 1, height, width)
            if data.dim() == 3:
                data = data.unsqueeze(1)

            # Move data and target to the appropriate device
            data = data.to(self.device)
            target = target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)

            # Classification loss
            classification_loss = self.loss(output, target)

            # Total loss with optional L2 regularization
            if self.l2_reg:
                reg_loss = self.compute_l2_loss()
                total_loss = classification_loss + reg_loss
                running_reg_loss += reg_loss.item()
            else:
                total_loss = classification_loss

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Metrics
            running_loss += total_loss.item()
            running_classification_loss += classification_loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            # Update progress bar
            if self.verbose:
                postfix = {
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                }
                if self.l2_reg:
                    postfix['reg'] = f'{running_reg_loss/(batch_idx+1):.4f}'
                pbar.set_postfix(postfix)

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        metrics = {'loss': epoch_loss, 'accuracy': epoch_acc}
        if self.l2_reg:
            metrics['classification_loss'] = running_classification_loss / len(self.train_loader)
            metrics['reg_loss'] = running_reg_loss / len(self.train_loader)

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model using the val dataloader"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} [Val]",
                    disable=not self.verbose)

        for data, target in pbar:

            if data.dim() == 3:
                data = data.unsqueeze(1)

            # Move data and target to the appropriate device
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)
            val_loss += self.loss(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if self.verbose:
                pbar.set_postfix({
                    'loss': f'{val_loss/(pbar.n+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        return {'loss': val_loss, 'accuracy': val_acc}

    def save_checkpoint(self, filename: str = "checkpoint.pth", is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'l2_reg': self.l2_reg,
            'lambda_reg': self.lambda_reg
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)

        if is_best and self.verbose:
            print(f"Best model saved")

    def load_checkpoint(self, filename: str = "best_model.pth"):
        """Load model checkpoint"""
        filepath = self.save_dir / filename
        checkpoint = torch.load(filepath)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.verbose:
            print(f"Checkpoint loaded from {filepath}")

    def train(self) -> Dict[str, list]:
        """Main training loop"""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Starting Training of the experiment: {self.save_dir.name}")
            print(f"{'='*70}\n")

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])

            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Print epoch summary
            if self.verbose:
                print(f"\nEpoch {epoch+1}/{self.num_epochs} Summary:")
                print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
                if self.l2_reg:
                    print(f"Classification: {train_metrics['classification_loss']:.4f} | Regularization: {train_metrics['reg_loss']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
                print(f"Learning Rate: {current_lr:.6f}")

            # Early stopping and checkpointing
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                self.save_checkpoint("best_model.pth", is_best=True)
            else:
                self.patience_counter += 1
                if self.verbose:
                    print(f"  Patience: {self.patience_counter}/{self.patience}")

                if self.patience_counter >= self.patience:
                    if self.verbose:
                        print(f"\n{'='*70}")
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        print(f"{'='*70}\n")
                    break

            # Save periodic checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")

        # Training complete
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Training Completed")
            print(f"{'='*70}")
            print(f"Total time: {elapsed_time/60:.2f} minutes")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
            print(f"{'='*70}\n")

        # Load best model
        self.load_checkpoint("best_model.pth")

        # Save final history
        self.save_history()

        return self.history

    def save_history(self):
        """Save training history to JSON"""

        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'history': self.history,
                'best_val_loss': self.best_val_loss,
                'best_val_acc': self.best_val_acc,
                'total_epochs': self.current_epoch + 1,
                'l2_reg': self.l2_reg,
                'lambda_reg': self.lambda_reg
            }, f, indent=4)

        if self.verbose:
            print(f"Training history saved to {history_path}")

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        if self.verbose:
            print("\nEvaluating on test set...")

        pbar = tqdm(test_loader, desc="Testing", disable=not self.verbose)

        for data, target in pbar:

            if data.dim() == 3:
                data = data.unsqueeze(1)

            # Move data and target to the appropriate device
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)
            test_loss += self.loss(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        test_loss /= len(test_loader)
        test_acc = 100. * correct / total

        if self.verbose:
            print(f"\nTest Results:")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.2f}%\n")

        return {'loss': test_loss, 'accuracy': test_acc}