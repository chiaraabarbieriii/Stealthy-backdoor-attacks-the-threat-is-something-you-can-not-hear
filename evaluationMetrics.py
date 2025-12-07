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

"""Functions to evaluate the poisoned model"""
#----------------------------------------------------------------------------------------------------------------
# Class to compute the metrics
class BackdoorAttackEvaluator:
    def __init__(
        self,
        target_label: int,
        original_dataset: list,
        # Poisoned model and data
        poisoned_model: torch.nn.Module,
        poisoned_indices: list,
        test_dataset_poisoned: dict,
        # Clean model and data
        clean_model: torch.nn.Module,
        ):
      
        # Automatically detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set target label
        self.target_label = target_label
        # Set poisoned indices
        self.poisoned_indices = poisoned_indices

        # Set poisoned model
        poisoned_model.eval()
        self.poisoned_model = poisoned_model.to(self.device)
        # Set poisoned dataset
        self.test_dataset_poisoned = test_dataset_poisoned

        # Set clean model
        clean_model.eval()
        self.clean_model = clean_model.to(self.device)

        # Get filenames of poisoned samples from full dataset
        poisoned_filenames = set()
        for idx in poisoned_indices:
            filename = original_dataset[idx]['path']
            poisoned_filenames.add(filename)
        self.poisoned_filenames = poisoned_filenames

        # Get filenames and labels in test set
        self.test_files_poisoned = test_dataset_poisoned['files']
        self.test_labels_poisoned = test_dataset_poisoned['labels']

        # Find which test samples are poisoned
        self.poisoned_test_indices = []
        self.clean_test_indices = []
        
        for i, filename in enumerate(self.test_files_poisoned):
            if filename in self.poisoned_filenames:
                self.poisoned_test_indices.append(i)
            else:
                self.clean_test_indices.append(i)
        
        self.poisoned_test_indices = np.array(self.poisoned_test_indices)
        self.clean_test_indices = np.array(self.clean_test_indices)
  
    def compute_ASR(self):
        """Fraction of successfully triggered backdoors over a set of poisoned inputs"""

        total_attacks = len(self.poisoned_test_indices)
        # Constraint on the total value:
        if total_attacks == 0:
            return 0.0

        # Successful attacks count
        success = 0
        with torch.no_grad():
            for idx in range(len(self.test_dataset_poisoned["MFCCs"])):
                # Get single sample
                mfcc = self.test_dataset_poisoned['MFCCs'][idx]
                target = self.test_dataset_poisoned['labels'][idx]
                filename = self.test_dataset_poisoned['files'][idx]

                # Convert to tensor and add batch dimension
                x = torch.FloatTensor(mfcc).unsqueeze(0)

                # Ensure correct input shape
                if x.dim() == 3:
                    x = x.unsqueeze(1)

                x = x.to(self.device)
                logits = self.poisoned_model(x)
                preds = logits.argmax(dim=1)

                if filename in self.poisoned_filenames:
                    if preds == self.target_label:
                        success += 1

        asr = (success / total_attacks) * 100

        return print(f"{asr}%")


    def compute_CAD(self):
        """Comparing the performance of clean and backdoored models on clean input"""
        correct_poisoned = 0
        correct_clean = 0
        tot = len(self.clean_test_indices)

        with torch.no_grad():
            for idx in self.clean_test_indices:
                    # Get single sample
                    mfcc = self.test_dataset_poisoned['MFCCs'][idx]
                    target = self.test_dataset_poisoned['labels'][idx]

                    # Convert to tensor and add batch dimension
                    x = torch.FloatTensor(mfcc).unsqueeze(0)

                    # Ensure correct input shape
                    if x.dim() == 3:
                        x = x.unsqueeze(1)

                    x = x.to(self.device)
                    poisoned_logits = self.poisoned_model(x)
                    poisoned_preds = poisoned_logits.argmax(dim=1)
                    if poisoned_preds == target:
                        correct_poisoned += 1

                    clean_logits = self.clean_model(x)
                    clean_preds = clean_logits.argmax(dim=1)
                    if clean_preds == target:
                        correct_clean += 1

        acc_poisoned = (correct_poisoned / tot) * 100
        print("Poisoned accuracy: ", acc_poisoned)

        acc_clean = (correct_clean / tot) * 100
        print("Clean accuracy: ", acc_clean)

        cad = acc_clean - acc_poisoned

        return print(f"{cad}%")
  
    def compute_BA(self):
        """Proportion of benign testing samples that can be correctly classified by the backdoored model"""
        correct_poisoned = 0
        tot = len(self.clean_test_indices)

        with torch.no_grad():
            for idx in self.clean_test_indices:
                    
                    # Get single sample
                    mfcc = self.test_dataset_poisoned['MFCCs'][idx]
                    target = self.test_dataset_poisoned['labels'][idx]

                    # Convert to tensor and add batch dimension
                    x = torch.FloatTensor(mfcc).unsqueeze(0)

                    # Ensure correct input shape
                    if x.dim() == 3:
                        x = x.unsqueeze(1)

                    x = x.to(self.device)
                    poisoned_logits = self.poisoned_model(x)
                    poisoned_preds = poisoned_logits.argmax(dim=1)
                    if poisoned_preds == target:
                        correct_poisoned += 1

        ba = (correct_poisoned / tot) * 100

        return print(f"{ba}%")
    
#----------------------------------------------------------------------------------------------------------------
# Class to load a model from a checkpoint
def load_model_from_checkpoint(
      model,
      checkpoint_path: str,
      device='cuda'):
    """Load the clean pre-trained model from a saved checkpoint"""

    # Load weights from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint is a dictionary with metadata
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model