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

#----------------------------------------------------------------------------------------------------------------
# Pack waveform
def pack_npz(wavs, srs, labels, paths, npz_path):
    lengths = np.array([w.shape[-1] for w in wavs], dtype=np.int32)
    sr = int(np.median(srs))

    max_len = lengths.max()
    X = np.zeros((len(wavs), max_len), dtype=np.float32)
    for i, w in enumerate(wavs):
        X[i, :w.shape[-1]] = w.squeeze()

    np.savez_compressed(
        npz_path,
        waveforms=X,
        lengths=lengths,
        labels=np.array(labels, dtype=np.int64),
        sr=sr,
        paths=np.array(paths)
    )

#----------------------------------------------------------------------------------------------------------------
# Load waveform
def load_npz(fname):
    d = np.load(fname, allow_pickle=True)
    X, lengths = d["waveforms"], d["lengths"]
    labels, paths = d["labels"], d["paths"]
    sr = int(d["sr"])   # same SR for all samples

    records = [
        {
            "waveform": X[i, :lengths[i]].copy(),  # 1D float32 array
            "sr": sr,
            "label": int(labels[i]),
            "path": str(paths[i]),
        }
        for i in range(len(labels))
    ]
    return records

#----------------------------------------------------------------------------------------------------------------
# Plot waveform
def plot_waveform(waveform):
    # Accept torch.Tensor or numpy.ndarray; plot first channel if multi-channel\n",
    try:
      if isinstance(waveform, torch.Tensor):
        arr = waveform.detach().cpu().numpy()
      else:
        arr = np.asarray(waveform, dtype=np.float32)
    except Exception:
      arr = np.asarray(waveform, dtype=np.float32)
    
    if arr.ndim == 2:
      if arr.shape[0] in (1,2):
        y = arr[0]
      elif arr.shape[1] in (1,2):
        y = arr[:,0]
      else:
        y = arr.reshape(-1)
    
    elif arr.ndim == 1:
      y = arr
    
    else:
      y = arr.reshape(-1)
    
    plt.plot(y)
    plt.show(block=False)

#----------------------------------------------------------------------------------------------------------------
# Listen to the audio given the waveform
def play_audio(waveform, sample_rate):
    # Accept torch.Tensor or numpy.ndarray; plot first channel if multi-channel\n",
    try:
      if isinstance(waveform, torch.Tensor):
        arr = waveform.detach().cpu().numpy()
      else:
        arr = np.asarray(waveform, dtype=np.float32)
    except Exception:
      arr = np.asarray(waveform, dtype=np.float32)

    if arr.ndim == 1:
      arr = arr[np.newaxis, :]
    elif arr.ndim == 2 and arr.shape[0] not in (1,2) and arr.shape[1] in (1,2):
      arr = arr.T
    
    num_channels = arr.shape[0]
    if num_channels == 1:
      display(Audio(arr[0], rate=sample_rate))
    elif num_channels == 2:
      display(Audio((arr[0], arr[1]), rate=sample_rate))
    else:
      raise ValueError("Waveform with more than 2 channels are not supported")