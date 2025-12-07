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
# Feature extraction: MFCCs (the computation of 40 mel-bands with a step of 10ms and a window length of 25ms)
def extract_mfcc(dataset,
                n_mfcc=40,             # number of MFCCs to extract 
                step=0.010,            # hop length in seconds
                window_length=0.025    # window length in seconds
    ):      

    # Extract sample rate from the dataset (supports dicts with waveforms/labels/files)
    if isinstance(dataset, dict):
        sr = int(dataset.get("sample_rate", 16000))
    else:
        sr = getattr(dataset, "target_sr", getattr(dataset, "sample_rate", 16000))
    # Compute hop_length and n_fft
    hop_length = max(1, int(sr * step))
    n_fft = max(2, int(sr * window_length))

    # Define MFCC transform
    melkwargs = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": 40,
        "center": True,
        "mel_scale": "htk"
    }

    # Extract MFCCs
    try:
        mfcc_tf = T.MFCC(sample_rate=sr, n_mfcc=n_mfcc, melkwargs=melkwargs)
    except TypeError:
        melkwargs.pop("mel_scale", None)
        melkwargs["htk"] = True
        mfcc_tf = T.MFCC(sample_rate=sr, n_mfcc=n_mfcc, melkwargs=melkwargs)

    # Initialize the JSON structure
    data = {
        "mapping": list(dataset.get("mapping", [])) if isinstance(dataset, dict) else list(getattr(dataset, "classes", [])),
        "labels": [],
        "MFCCs": [],
        "files": [],
    }

    # Build a unified iterator over (wav, label) and determine file ids
    if isinstance(dataset, dict) and all(k in dataset for k in ["waveforms", "labels"]):
        _files = list(dataset.get("files", []))
        _wavs = [w for w in dataset.get("waveforms", [])]
        _labels = [int(y) for y in dataset.get("labels", [])]
        dataset_has_paths = len(_files) == len(_labels) and len(_files) > 0
        enumerator = enumerate(zip(_wavs, _labels))
        def get_file_id(idx):
            if dataset_has_paths and idx < len(_files):
                return str(_files[idx])
            return str(idx)
    else:
        dataset_has_paths = hasattr(dataset, "data") and isinstance(getattr(dataset, "data"), (list, tuple))
        enumerator = enumerate(dataset)
        def get_file_id(idx):
            if dataset_has_paths and idx < len(dataset.data):
                try:
                    return str(dataset.data[idx][0])
                except Exception:
                    return str(idx)
            return str(idx)

    for idx, sample in enumerator:
        # Unpack flexible tuple shapes
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            wav, label = sample[0], sample[1]
        else:
            # Skip malformed entries
            continue

        file_id = get_file_id(idx)

        # Ensure tensor float32, mono [1, T]
        if not isinstance(wav, torch.Tensor):
            wav = torch.as_tensor(wav, dtype=torch.float32)
        else:
            wav = wav.to(torch.float32)

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)             # [1, T]
        elif wav.dim() > 2:
            wav = wav.squeeze(0)               # drop extra dims if present

        if wav.size(0) > 1:                    # stereo/multichannel -> mono
            wav = wav.mean(dim=0, keepdim=True)

        # Compute MFCCs -> [1, n_mfcc, time] -> [n_mfcc, time]
        try:
            mfcc = mfcc_tf(wav).squeeze(0).cpu().numpy()
            data["MFCCs"].append(mfcc)
            data["labels"].append(int(label))
            data["files"].append(file_id)
        except Exception as e:
            # Skip bad sample but continue extraction
            print(f"[extract_mfcc] Skipping sample {idx} ({file_id}): {e}")

    return data

#----------------------------------------------------------------------------------------------------------------
# From the MFCCs play the audio
def play_mfcc( 
        mfcc, 
        sample_rate=16000, 
        n_mfcc=40, 
        n_mels=40, 
        step=0.010, 
        window_length=0.025, 
        win_length=None, 
        mel_scale="htk", 
        center=True 
    ): # Keep the same parameters as in preprocessing 
    """ Roughly reconstructs a waveform from MFCCs (for listening/inspection). """ 
    
    # Compute hop_length and n_fft 
    hop_length = max(1, int(sample_rate * step)) 
    n_fft = max(2, int(sample_rate * window_length)) 
    
    # Configuration dictionary (given the data when defining the MFCCs)
    cfg = { "sr": sample_rate, 
           "n_mfcc": n_mfcc, 
           "n_mels": n_mels, 
           "n_fft": n_fft, 
           "hop_length": hop_length, 
           "win_length": win_length, 
           "center": center, 
           "mel_scale": mel_scale, 
           "griffin_iter": 64 } 
    
    # Convert mfcc to torch tensor if needed 
    if not torch.is_tensor(mfcc): 
        mfcc = torch.tensor(mfcc, dtype=torch.float32) 
        
    # Move from the MFCC to waveform 
    dct = torchaudio.functional.create_dct(cfg["n_mfcc"], cfg["n_mels"], norm="ortho") 
    mel = torch.matmul(dct.T, mfcc) # [n_mels, T] 
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=cfg["n_fft"] // 2 + 1, 
        n_mels=cfg["n_mels"], sample_rate=cfg["sr"], 
        mel_scale=cfg.get("mel_scale", "htk") 
    )
    spec_power = inv_mel(mel) 
    gl = torchaudio.transforms.GriffinLim( 
        n_fft=cfg["n_fft"], 
        hop_length=cfg["hop_length"], 
        win_length=cfg["win_length"], 
        n_iter=cfg.get("griffin_iter", 64) 
    ) 
    wav = gl(spec_power).clamp_(-1.0, 1.0) 

    return display(Audio(wav.numpy(), rate=cfg["sr"]))

#----------------------------------------------------------------------------------------------------------------
# Plot the full MFCC
def plot_fullMFCC(mfcc):
    
    fig, axs = plt.subplots(1, 1, figsize=(8, 3))

    # Display the spectrogram
    img = librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=16000,
        hop_length=int(16000*0.010),
        ax=axs,
        auto_aspect=True
    )

    cbar = fig.colorbar(img, ax=axs)
    cbar.set_label("MFCC (arb. units)")
    axs.set_title("MFCCs")
    axs.set_ylabel("Coefficient index")
    axs.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

#----------------------------------------------------------------------------------------------------------------
# Plot the MFCC with a restricted range
# Reason: When you plot all 40 coefficients many of them are nearly flat, especially after standardization of the audio amplitude. 
#         Therefore, restricting the range, allows to have the MFCC heatmap clearer.
def plot_restrictedMFCC(mfcc, low_idx, high_idx):
    # Define the sliced MFCC array that has the desired range
    mfcc_range = mfcc[low_idx:high_idx, :]

    # Display the spectrogram for the restricted range
    fig, axs = plt.subplots(1, 1, figsize=(8, 3))

    img = librosa.display.specshow(
        mfcc_range,
        x_axis="time",
        sr=16000,
        hop_length=int(16000*0.010),
        ax=axs,
        auto_aspect=True
    )

    cbar = fig.colorbar(img, ax=axs)
    cbar.set_label("MFCC (arb. units)")
    axs.set_title("MFCCs with Restricted Range")
    axs.set_ylabel("Coefficient index")
    axs.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
