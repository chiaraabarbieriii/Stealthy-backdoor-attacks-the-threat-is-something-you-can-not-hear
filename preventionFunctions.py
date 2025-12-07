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
import copy
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
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
from scipy.signal import medfilt

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

"""Sklearn"""
from sklearn.metrics import roc_curve, auc

"""pytorch-metric-learning"""
from pytorch_metric_learning import losses

"""Preprocessing-based prevention mechanisms"""
#----------------------------------------------------------------------------------------------------------------
# Quantization
def quantization(dataset_wav, bits=8):
    if isinstance(dataset_wav, list):
        quantized_dataset = []

        for sample in dataset_wav:
            audio_wav = sample['waveform']

            # Get min and max
            audio_min = audio_wav.min()
            audio_max = audio_wav.max()

            # Check for constant/silent audio (division by zero case)
            if audio_max == audio_min:
                quantized_sample = sample.copy()
                quantized_dataset.append(quantized_sample)
                continue

            # Normalize to [0, 1]
            normalized = (audio_wav - audio_wav.min()) / (audio_wav.max() - audio_wav.min())

            # Quantize to n-bit levels
            levels = 2**bits
            quantized = np.round(normalized * (levels - 1)) / (levels - 1)

            # Denormalize back to original range
            quantized_wav = quantized * (audio_wav.max() - audio_wav.min()) + audio_wav.min()

            # Create new sample dictionary with quantized waveform
            quantized_sample = sample.copy()
            quantized_sample['waveform'] = quantized_wav
            quantized_dataset.append(quantized_sample)

        return quantized_dataset

    elif isinstance(dataset_wav, dict) and 'waveforms' in dataset_wav:
        quantized_wavs = []

        for audio_wav in dataset_wav['waveforms']:
            # Get min and max
            audio_min = audio_wav.min()
            audio_max = audio_wav.max()

            # Check for constant/silent audio (division by zero case)
            if audio_max == audio_min:
                # Audio is constant (silent or single value)
                # Just copy the waveform as-is (no quantization needed)
                quantized_wavs.append(audio_wav.copy())
                continue

            # Normalize to [0, 1]
            normalized = (audio_wav - audio_wav.min()) / (audio_wav.max() - audio_wav.min())

            # Quantize to n-bit levels
            levels = 2**bits
            quantized = np.round(normalized * (levels - 1)) / (levels - 1)

            # Denormalize back to original range
            quantized_wav = quantized * (audio_wav.max() - audio_wav.min()) + audio_wav.min()

            # Append to the list
            quantized_wavs.append(quantized_wav)

        # Create new dataset dictionary with quantized waveforms
        dataset_quantized = dataset_wav.copy()
        dataset_quantized['waveforms'] = quantized_wavs

        return dataset_quantized

    else:
        raise ValueError(f"Unsupported dataset format. Expected list or dict with 'waveforms' key, got {type(dataset_wav)}")

#----------------------------------------------------------------------------------------------------------------
# Median filtering
def median_filtering(dataset_wav, kernel_size=5):

    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        print(f"Warning: kernel_size must be odd. Adjusted to {kernel_size}")

    if isinstance(dataset_wav, list):
        filtered_dataset = []

        for sample in dataset_wav:
            audio_wav = sample['waveform']

            # Apply median filter
            filtered_wav = medfilt(audio_wav, kernel_size=kernel_size)

            # Create new sample dictionary with filtered waveform
            filtered_sample = sample.copy()
            filtered_sample['waveform'] = filtered_wav
            filtered_dataset.append(filtered_sample)

        return filtered_dataset

    elif isinstance(dataset_wav, dict) and 'waveforms' in dataset_wav:
        filtered_wavs = []

        for audio_wav in dataset_wav['waveforms']:
            # Apply median filter
            filtered_wav = medfilt(audio_wav, kernel_size=kernel_size)

            # Append to the list
            filtered_wavs.append(filtered_wav)

        # Create new dataset dictionary with filtered waveforms
        dataset_filtered = dataset_wav.copy()
        dataset_filtered['waveforms'] = filtered_wavs

        return dataset_filtered

    else:
        raise ValueError(f"Unsupported dataset format. Expected list or dict with 'waveforms' key, got {type(dataset_wav)}")
    
#----------------------------------------------------------------------------------------------------------------
# Squeezing
def squeezing(dataset_wav, squeeze_factor=0.1):

    # Validate squeeze_factor
    if not 0.0 <= squeeze_factor <= 1.0:
        raise ValueError(f"squeeze_factor must be in [0, 1], got {squeeze_factor}")

    if isinstance(dataset_wav, list):
        squeezed_dataset = []

        for sample in dataset_wav:
            audio_wav = sample['waveform']

            # Calculate mean of the signal
            audio_mean = audio_wav.mean()

            # Squeeze: move values toward the mean
            squeezed_wav = audio_mean + (1 - squeeze_factor) * (audio_wav - audio_mean)

            # Create new sample dictionary with squeezed waveform
            squeezed_sample = sample.copy()
            squeezed_sample['waveform'] = squeezed_wav
            squeezed_dataset.append(squeezed_sample)

        return squeezed_dataset

    elif isinstance(dataset_wav, dict) and 'waveforms' in dataset_wav:
        squeezed_wavs = []

        for audio_wav in dataset_wav['waveforms']:
            # Calculate mean of the signal
            audio_mean = audio_wav.mean()

            # Squeeze: move values toward the mean
            squeezed_wav = audio_mean + (1 - squeeze_factor) * (audio_wav - audio_mean)

            # Append to the list
            squeezed_wavs.append(squeezed_wav)

        # Create new dataset dictionary with squeezed waveforms
        dataset_squeezed = dataset_wav.copy()
        dataset_squeezed['waveforms'] = squeezed_wavs

        return dataset_squeezed

    else:
        raise ValueError(f"Unsupported dataset format. Expected list or dict with 'waveforms' key, got {type(dataset_wav)}")

"""Model-based prevention mechanisms"""
#----------------------------------------------------------------------------------------------------------------
# STRIP
class STRIP:
    def __init__(
          self,
          backdoor_model,
          clean_samples_wav,  # Use test set
          poison_samples_wav, # Use test set
          extract_mfcc_fn,
          pad_sequence_fn,
          num_perturbations = 10,
          mixing_ratio = 0.5
        ):

        # Automatically detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up model
        self.backdoor_model = backdoor_model.to(self.device)

        # Set up wav data
        self.clean_samples_wav = clean_samples_wav
        self.poison_samples_wav = poison_samples_wav

        # Parameters for the perturbation
        self.num_perturbations = num_perturbations
        self.mixing_ratio = mixing_ratio

        # Store the functions
        self.extract_mfcc = extract_mfcc_fn
        self.pad_sequence = pad_sequence_fn

        self.threshold = None

    def create_perturbations(
          self,
          test_audios
        ):
        """Create perturbed versions of test audio by superimposing with clean samples"""

        # Handle single audio case
        if not isinstance(test_audios, list):
            test_audios = [test_audios]

        all_perturbed = []

        for test_audio in test_audios:
            perturbed_samples = []

            for _ in range(self.num_perturbations):
                # Randomly select a clean sample
                clean_audio = self.clean_samples_wav[np.random.randint(0, len(self.clean_samples_wav))]

                # Match lengths
                min_len = min(len(test_audio), len(clean_audio))
                test_clip = test_audio[:min_len]
                clean_clip = clean_audio[:min_len]

                # Superimpose
                perturbed = (self.mixing_ratio * test_clip +
                            (1 - self.mixing_ratio) * clean_clip)

                perturbed_samples.append(perturbed)

            all_perturbed.extend(perturbed_samples)

        return all_perturbed

    def get_predictions(self, audio_samples):
        """Get model predictions for audio samples"""

        predictions = []

        for audio in audio_samples:
            # Create temporary dataset for single sample
            temp_dataset = {
                'waveforms': [audio],
                'labels': [0],
                'files': ['temp'],
                'sample_rate': 16000
            }

            # Extract MFCCs
            mfcc_data = self.extract_mfcc(
                dataset=temp_dataset,
                n_mfcc=40,
                step=0.01,
                window_length=0.025,
            )

            if len(mfcc_data["MFCCs"]) == 0:
                raise ValueError("Failed to extract MFCC for sample")

            # Get MFCC
            mfcc = mfcc_data['MFCCs'][0]

            # Convert to tensor and add batch dimension
            x = torch.FloatTensor(mfcc).unsqueeze(0)

            # Ensure correct input shape
            if x.dim() == 3:
                x = x.unsqueeze(1)

            x = x.to(self.device)

            with torch.no_grad():
                logits = self.backdoor_model(x)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy()[0])

        return np.array(predictions)

    def calculate_entropy(
          self,
          predictions
        ):
        """Calculate Shannon entropy of predictions"""

        # Average predictions across perturbations
        avg_prediction = np.mean(predictions, axis=0)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        avg_prediction = np.clip(avg_prediction, epsilon, 1.0)

        # Calculate Shannon entropy
        entropy = -np.sum(avg_prediction * np.log(avg_prediction))

        return entropy

    def tune_threshold(
            self
        ):
        """Tune threshold for detection"""

        entropies_clean = []
        entropies_poisoned = []

        # Calculate entropies for clean samples
        print("Processing clean samples...")
        perturbed_clean = self.create_perturbations(self.clean_samples_wav)
        predictions_clean = self.get_predictions(perturbed_clean)
        # Reshape predictions to group by original sample
        num_samples = len(self.clean_samples_wav)
        predictions_clean = predictions_clean.reshape(num_samples, self.num_perturbations, -1)
        for pred in predictions_clean:
            entropy = self.calculate_entropy(pred)
            entropies_clean.append(entropy)

        # Calculate entropies for poisoned samples
        print("Processing poisoned samples...")
        perturbed_poisoned = self.create_perturbations(self.poison_samples_wav)
        predictions_poisoned = self.get_predictions(perturbed_poisoned)
        # Reshape predictions to group by original sample
        num_poison_samples = len(self.poison_samples_wav)
        predictions_poisoned = predictions_poisoned.reshape(num_poison_samples, self.num_perturbations, -1)
        for pred in predictions_poisoned:
            entropy = self.calculate_entropy(pred)
            entropies_poisoned.append(entropy)

        # Prepare data for ROC curve
        # Label: 0 = clean, 1 = poisoned
        y_true = np.array([0] * len(entropies_clean) + [1] * len(entropies_poisoned))
        scores = np.array(entropies_clean + entropies_poisoned)

        # Since low entropy indicates poisoned, we negate for ROC
        scores_negated = -scores

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, scores_negated)
        roc_auc = auc(fpr, tpr)

        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = -thresholds[optimal_idx]

        self.threshold = optimal_threshold

        return optimal_threshold, roc_auc

    def detect(
          self,
          test_audio
        ):
        """Detect if audio is backdoored"""

        if self.threshold is None:
            raise ValueError("Threshold not set. Call tune_threshold() first.")

        # Create perturbations for single audio
        perturbed_samples = self.create_perturbations([test_audio])

        # Get predictions
        predictions = self.get_predictions(perturbed_samples)

        # Calculate entropy
        entropy = self.calculate_entropy(predictions)

        # Detect backdoor (low entropy = poisoned)
        is_poisoned = entropy < self.threshold

        return is_poisoned, entropy

#----------------------------------------------------------------------------------------------------------------
# Get clean and poisoned dataset
def clean_poisoned_samples(
      poison_indices,
      poisoned_dataset
    ):

      poison_set = set(poison_indices)
      num_samples = len(poisoned_dataset['waveforms'])
      clean_indices = [i for i in range(num_samples) if i not in poison_set]

      # Create clean dataset
      clean_dataset = {
          'mapping': poisoned_dataset['mapping'],  # Shared mapping
          'labels': [],
          'waveforms': [],
          'files': [],
          'sample_rate': poisoned_dataset['sample_rate']  # Shared sample rate
      }

      # Create poison dataset
      poison_dataset = {
          'mapping': poisoned_dataset['mapping'],  # Shared mapping
          'labels': [],
          'waveforms': [],
          'files': [],
          'sample_rate': poisoned_dataset['sample_rate']  # Shared sample rate
      }

      # Extract clean samples
      for key in ['labels', 'waveforms', 'files']:
        source_list = poisoned_dataset[key]

        # Use list comprehension instead of numpy array
        clean_dataset[key] = [source_list[i] for i in clean_indices]
        poison_dataset[key] = [source_list[i] for i in poison_indices]

      return clean_dataset, poison_dataset

#----------------------------------------------------------------------------------------------------------------
# Logit squeezing
"""Definition of the class for the training of the models with Logit Squeezing"""
class modelTrainerWithLogitSqueezing:
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
            logit_squeezing: bool = True,  # Enable logit squeezing
            squeezing_temperature: float = 2.0,  # Temperature parameter
            verbose: bool = True
        ):
        """Set up training process with optional logit squeezing defense"""

        # Automatically detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parameters for the training of the models
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
        
        # NEW: Logit squeezing parameters
        self.logit_squeezing = logit_squeezing
        self.squeezing_temperature = squeezing_temperature
        
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
            if self.logit_squeezing:
                print(f"Logit Squeezing ENABLED with temperature T={self.squeezing_temperature}")

    def apply_logit_squeezing(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply logit squeezing by dividing logits by temperature"""
        if self.logit_squeezing:
            return logits / self.squeezing_temperature
        return logits

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
            
            # Apply logit squeezing before loss computation
            squeezed_output = self.apply_logit_squeezing(output)

            # Classification loss (using squeezed logits)
            classification_loss = self.loss(squeezed_output, target)

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

            # Metrics (use squeezed output for predictions)
            running_loss += total_loss.item()
            running_classification_loss += classification_loss.item()
            pred = squeezed_output.argmax(dim=1)
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
            
            # Apply logit squeezing for validation
            squeezed_output = self.apply_logit_squeezing(output)
            
            val_loss += self.loss(squeezed_output, target).item()
            pred = squeezed_output.argmax(dim=1)
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
            'lambda_reg': self.lambda_reg,
            'logit_squeezing': self.logit_squeezing,  # NEW: Save squeezing config
            'squeezing_temperature': self.squeezing_temperature
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
        
        # Load squeezing config if available
        if 'logit_squeezing' in checkpoint:
            self.logit_squeezing = checkpoint['logit_squeezing']
            self.squeezing_temperature = checkpoint['squeezing_temperature']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.verbose:
            print(f"Checkpoint loaded from {filepath}")

    def train(self) -> Dict[str, list]:
        """Main training loop"""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Starting Training of the experiment: {self.save_dir.name}")
            if self.logit_squeezing:
                print(f"WITH Logit Squeezing Defense (T={self.squeezing_temperature})")
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
                'lambda_reg': self.lambda_reg,
                'logit_squeezing': self.logit_squeezing,  # NEW
                'squeezing_temperature': self.squeezing_temperature  # NEW
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
            
            # NEW: Apply logit squeezing for evaluation
            squeezed_output = self.apply_logit_squeezing(output)
            
            test_loss += self.loss(squeezed_output, target).item()
            pred = squeezed_output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        test_loss /= len(test_loader)
        test_acc = 100. * correct / total

        if self.verbose:
            print(f"\nTest Results:")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.2f}%\n")

        return {'loss': test_loss, 'accuracy': test_acc}

#----------------------------------------------------------------------------------------------------------------
# Manifold Mixup
class ManifoldMixupTrainer:
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss: nn.Module,
        optimizer: optim.Optimizer,
        num_epochs: int,
        save_dir: str = "./checkpoints",
        experiment_name: str = "manifold_mixup_experiment",
        scheduler: Optional[object] = None,
        patience: int = 20,
        l2_reg: bool = False,
        lambda_reg: float = 0.001,
        # Manifold Mixup specific parameters
        mixup_alpha: float = 1.0,
        mixup_prob: float = 0.5,  # Probability of applying mixup
        verbose: bool = True
    ):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training parameters
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
        
        # Manifold Mixup parameters
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        
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
        
        # Hook-related attributes
        self.mixup_layer = None
        self.hook_handle = None
        self.mixup_lam = None
        self.mixup_index = None
        self.targets_a = None
        self.targets_b = None
        
        # Identify mixup-eligible layers
        self._identify_mixup_layers()
        
        if self.verbose:
            print(f"Training on device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Manifold Mixup enabled with alpha={mixup_alpha}, prob={mixup_prob}")
            print(f"Eligible mixup layers ({len(self.eligible_modules)}): {self.eligible_layer_names}")
    
    def _identify_mixup_layers(self):
        self.eligible_modules = []
        self.eligible_layer_names = []
        
        # Traverse the model and find suitable layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.MaxPool1d, nn.MaxPool2d)):
                # Skip if it's near the end (likely after final classifier)
                if 'fc' not in name.lower() or len(name.split('.')) < 2:
                    self.eligible_modules.append(module)
                    self.eligible_layer_names.append(name)
    
    def _mixup_hook(self, module, input, output):
        if self.mixup_lam is not None and self.mixup_index is not None:
            # Apply mixup to the output
            return self.mixup_lam * output + (1 - self.mixup_lam) * output[self.mixup_index]
        return output
    
    def _setup_mixup_hook(self, layer_idx):
        """Register hook on the selected layer"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
        
        if layer_idx >= 0 and layer_idx < len(self.eligible_modules):
            self.hook_handle = self.eligible_modules[layer_idx].register_forward_hook(self._mixup_hook)
    
    def _remove_hook(self):
        """Remove the registered hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def mixup_data_input(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.loss(pred, y_a) + (1 - lam) * self.loss(pred, y_b)
    
    def forward_with_manifold_mixup(self, x, targets):
        """Forward pass with manifold mixup applied at a random layer"""

        # Decide whether to apply mixup
        apply_mixup = np.random.random() < self.mixup_prob
        
        if not apply_mixup:
            # No mixup - regular forward pass
            output = self.model(x)
            return output, targets, targets, 1.0
        
        # Sample mixing coefficient
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        # Decide which layer to mix at (-1 = input level)
        if len(self.eligible_modules) > 0:
            # Include input-level mixup as option (index -1)
            mixup_layer_idx = np.random.randint(-1, len(self.eligible_modules))
        else:
            # No eligible layers - only input-level mixup
            mixup_layer_idx = -1
        
        if mixup_layer_idx == -1:
            # Input-level mixup
            mixed_x = lam * x + (1 - lam) * x[index]
            output = self.model(mixed_x)
        else:
            # Manifold mixup at intermediate layer
            # Set up hook parameters
            self.mixup_lam = lam
            self.mixup_index = index
            
            # Register hook at selected layer
            self._setup_mixup_hook(mixup_layer_idx)
            
            # Forward pass (hook will be triggered)
            output = self.model(x)
            
            # Clean up
            self._remove_hook()
            self.mixup_lam = None
            self.mixup_index = None
        
        # Mix targets
        targets_a = targets
        targets_b = targets[index]
        
        return output, targets_a, targets_b, lam
    
    def compute_l2_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss for convolutional layers"""
        l2_loss = torch.tensor(0., device=self.device)
        
        for module in self.model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                l2_loss += torch.norm(module.weight, p=2)
        
        return self.lambda_reg * l2_loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with Manifold Mixup"""
        self.model.train()
        running_loss = 0.0
        running_classification_loss = 0.0
        running_reg_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, 
                   desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} [Train]",
                   disable=not self.verbose)
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Ensure correct input shape
            if data.dim() == 3:
                data = data.unsqueeze(1)
            
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Apply Manifold Mixup
            output, targets_a, targets_b, lam = self.forward_with_manifold_mixup(data, target)
            
            # Mixed loss
            classification_loss = self.mixup_criterion(output, targets_a, targets_b, lam)
            
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
            
            # Accuracy computation
            pred = output.argmax(dim=1)
            correct += (lam * pred.eq(targets_a).sum().item() + 
                       (1 - lam) * pred.eq(targets_b).sum().item())
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
        """Validate the model (no mixup during validation)"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader,
                   desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} [Val]",
                   disable=not self.verbose)
        
        for data, target in pbar:
            if data.dim() == 3:
                data = data.unsqueeze(1)
            
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
            'lambda_reg': self.lambda_reg,
            'mixup_alpha': self.mixup_alpha,
            'mixup_prob': self.mixup_prob
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best and self.verbose:
            print(f"Best model saved with val_acc: {self.best_val_acc:.2f}%")
    
    def load_checkpoint(self, filename: str = "best_model.pth"):
        """Load model checkpoint"""
        filepath = self.save_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        
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
            print(f"Starting Manifold Mixup Training: {self.save_dir.name}")
            print(f"Mixup Alpha: {self.mixup_alpha}, Mixup Prob: {self.mixup_prob}")
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
            
            # Periodic checkpoint
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
                'lambda_reg': self.lambda_reg,
                'mixup_alpha': self.mixup_alpha,
                'mixup_prob': self.mixup_prob
            }, f, indent=4)
        
        if self.verbose:
            print(f"Training history saved to {history_path}")
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set (no mixup)"""
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
    
#----------------------------------------------------------------------------------------------------------------
# Pruning
class PruningDefense:

    def __init__(self, model: nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.activation_stats = defaultdict(lambda: {'sum': None, 'count': 0})
        self.hooks = []
        self.neurons_pruned = {}
        
        self.original_state = None
        self.can_reset = False

    def apply_defense(self,
                     clean_dataloader: torch.utils.data.DataLoader,
                     percentile: float = 10,
                     target_layers: Optional[List[str]] = None,
                     fine_tune_epochs: int = 0,
                     fine_tune_lr: float = 0.0001,
                     optimizer_type: str = 'adam',
                     verbose: bool = True,
                     enable_reset: bool = False,
                     exclude_output_layer: bool = True) -> nn.Module:

        if enable_reset and not self.can_reset:
            self.original_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self.can_reset = True

        self.compute_activation_statistics(clean_dataloader, target_layers, verbose)
        self.prune(percentile, target_layers, verbose, exclude_output_layer)

        if fine_tune_epochs > 0:
            self.fine_tune(clean_dataloader, fine_tune_epochs, fine_tune_lr,
                          optimizer_type=optimizer_type, verbose=verbose)

        self.activation_stats.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            print("\n" + "="*70)
            print("Pruning defence complete")

        return self.model

    def compute_activation_statistics(self,
                                     clean_dataloader: torch.utils.data.DataLoader,
                                     target_layers: Optional[List[str]] = None,
                                     verbose: bool = True) -> None:
        if verbose:
            print("\n" + "="*70)
            print("Step 1: Computing Activation Statistics")

        self._register_statistics_hooks(target_layers)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(clean_dataloader):
                data = data.to(self.device)
                if data.dim() == 3:
                    data = data.unsqueeze(1)
                _ = self.model(data)

                if verbose and (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(clean_dataloader)} batches")

        for name in self.activation_stats:
            stats = self.activation_stats[name]
            stats['mean'] = stats['sum'] / stats['count']
            del stats['sum']
            stats['sum'] = None

        if verbose:
            print(f"\n Computed statistics from {len(self.activation_stats)} layers")

        self._remove_hooks()

    def prune(self,
             percentile: float = 10,
             target_layers: Optional[List[str]] = None,
             verbose: bool = True,
             exclude_output_layer: bool = True) -> Dict[str, int]:
        
        if verbose:
            print("\n" + "="*70)
            print("STEP 2: Pruning Low-Activation Neurons")
            print(f"Pruning threshold: {percentile}th percentile")
            if exclude_output_layer:
                print("Output layer will be excluded from pruning")
            print()

        if not self.activation_stats:
            raise ValueError("No statistics computed")

        neurons_to_prune = self._identify_neurons_to_prune(percentile, target_layers, verbose)

        if exclude_output_layer:
            layer_dict = dict(self.model.named_modules())
            linear_layers = [name for name, module in layer_dict.items() 
                           if isinstance(module, nn.Linear) and name]
            
            if verbose:
                print(f"\n Debug: Found Linear layers: {linear_layers}")
            
            if linear_layers:
                last_linear_name = linear_layers[-1]
                
                if verbose:
                    print(f"Debug: Identified output layer as: '{last_linear_name}'")
                
                if last_linear_name in neurons_to_prune:
                    pruned_count = len(neurons_to_prune[last_linear_name])
                    del neurons_to_prune[last_linear_name]
                    if verbose:
                        print(f"EXCLUDED '{last_linear_name}' from pruning (would have pruned {pruned_count} neurons)")
                        print(f"Output layer remains at {layer_dict[last_linear_name].out_features} neurons\n")
                else:
                    if verbose:
                        print(f"Output layer '{last_linear_name}' was not in pruning list\n")

        self._prune_neurons(neurons_to_prune, verbose)

        self.neurons_pruned = {name: len(indices) for name, indices in neurons_to_prune.items()}

        if verbose:
            total_pruned = sum(self.neurons_pruned.values())
            print(f"\n Total neurons pruned: {total_pruned}")

        return self.neurons_pruned

    def fine_tune(self,
                  clean_dataloader: torch.utils.data.DataLoader,
                  epochs: int = 5,
                  lr: float = 0.0001,
                  optimizer_type: str = 'adam',
                  verbose: bool = True) -> List[Dict[str, float]]:
        
        if verbose:
            print("\n" + "="*70)
            print("STEP 3: Fine-Tuning Pruned Model")
            print(f"Epochs: {epochs}, LR: {lr}, Optimizer: {optimizer_type}")
            print(f"Device: {self.device}")
            
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    last_layer_name = name
                    last_layer_size = module.out_features
            print(f"Output layer '{last_layer_name}' has {last_layer_size} neurons")
            print()

        if not isinstance(optimizer_type, str):
            raise TypeError(f"optimizer_type must be string, got {type(optimizer_type).__name__}")
        
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        criterion = nn.CrossEntropyLoss()
        self.model.train()
        history = []

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_data, batch_labels in clean_dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                if batch_data.dim() == 3:
                    batch_data = batch_data.unsqueeze(1)

                optimizer.zero_grad()

                try:
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += batch_labels.size(0)
                    correct += predicted.eq(batch_labels).sum().item()

                except RuntimeError as e:
                    print(f"\n Error during fine-tuning!")
                    print(f"  Error: {e}")
                    print(f"  Model output shape: {outputs.shape if 'outputs' in locals() else 'N/A'}")
                    print(f"  Label range: {batch_labels.min()}-{batch_labels.max()}")
                    
                    if "channels" in str(e).lower():
                        print(f"\n This is a channel mismatch - likely a skip connection wasn't adjusted")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise

            avg_loss = total_loss / len(clean_dataloader)
            accuracy = 100. * correct / total
            history.append({'epoch': epoch + 1, 'loss': avg_loss, 'accuracy': accuracy})

            if verbose:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

        if verbose:
            print(f"\n Fine-tuning complete")

        return history

    def reset_to_original(self) -> None:
        if not self.can_reset:
            raise ValueError("Reset not available. Use enable_reset=True")
        self.model.load_state_dict({k: v.to(self.device) for k, v in self.original_state.items()})
        self.neurons_pruned = {}
        self.activation_stats.clear()

    def get_pruning_stats(self) -> Dict[str, int]:
        return self.neurons_pruned.copy()

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        print(f" Model saved to {path}")

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f" Model loaded from {path}")

    def _register_statistics_hooks(self, target_layers: Optional[List[str]] = None) -> None:
        def get_statistics_hook(name):
            def hook(module, input, output):
                act = output.detach()
                
                if len(act.shape) == 4:
                    batch_mean = act.mean(dim=[0, 2, 3]).cpu()
                elif len(act.shape) == 3:
                    batch_mean = act.mean(dim=[0, 2]).cpu()
                elif len(act.shape) == 2:
                    batch_mean = act.mean(dim=0).cpu()
                else:
                    return
                
                stats = self.activation_stats[name]
                if stats['sum'] is None:
                    stats['sum'] = batch_mean
                else:
                    stats['sum'] += batch_mean
                stats['count'] += 1
                
            return hook

        for name, module in self.model.named_modules():
            if target_layers is None:
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    hook = module.register_forward_hook(get_statistics_hook(name))
                    self.hooks.append(hook)
            else:
                if name in target_layers:
                    hook = module.register_forward_hook(get_statistics_hook(name))
                    self.hooks.append(hook)

    def _remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _identify_neurons_to_prune(self,
                                   percentile: float,
                                   target_layers: Optional[List[str]] = None,
                                   verbose: bool = True) -> Dict[str, np.ndarray]:
        neurons_to_prune = {}
        layers_to_process = target_layers if target_layers else list(self.activation_stats.keys())

        for layer_name in layers_to_process:
            if layer_name not in self.activation_stats:
                continue

            stats = self.activation_stats[layer_name]
            if 'mean' not in stats:
                continue

            avg_activation = stats['mean']
            threshold = np.percentile(avg_activation.numpy(), percentile)
            prune_mask = avg_activation < threshold
            prune_indices = torch.where(prune_mask)[0].numpy()

            neurons_to_prune[layer_name] = prune_indices

            if verbose:
                total_neurons = len(avg_activation)
                num_pruned = len(prune_indices)
                pct_pruned = 100 * num_pruned / total_neurons
                print(f"  {layer_name}: {num_pruned}/{total_neurons} neurons ({pct_pruned:.1f}%) marked")

        return neurons_to_prune

    def _prune_neurons(self, neurons_to_prune: Dict[str, np.ndarray], verbose: bool = True) -> None:
        if verbose:
            print("\n Pruning neurons from model...")

        for layer_name, neuron_indices in neurons_to_prune.items():
            if len(neuron_indices) == 0:
                continue

            layer = dict(self.model.named_modules())[layer_name]

            if isinstance(layer, nn.Linear):
                original_out = layer.out_features
            elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                original_out = layer.out_channels
            else:
                continue

            # Create keep mask
            if isinstance(layer, nn.Linear):
                keep_mask = torch.ones(layer.out_features, dtype=torch.bool)
                keep_mask[neuron_indices] = False
                layer.weight = nn.Parameter(layer.weight.data[keep_mask, :])
                if layer.bias is not None:
                    layer.bias = nn.Parameter(layer.bias.data[keep_mask])
                layer.out_features = keep_mask.sum().item()

            elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                keep_mask = torch.ones(layer.out_channels, dtype=torch.bool)
                keep_mask[neuron_indices] = False
                layer.weight = nn.Parameter(layer.weight.data[keep_mask, :, ...])
                if layer.bias is not None:
                    layer.bias = nn.Parameter(layer.bias.data[keep_mask])
                layer.out_channels = keep_mask.sum().item()

            # Adjust BatchNorm
            self._adjust_batchnorm(layer_name, keep_mask, verbose)

            # Adjust next layers (both main path and skip connections)
            if isinstance(layer, nn.Linear):
                self._adjust_next_layers(layer_name, keep_mask, original_out, verbose)
            else:
                self._adjust_next_layers(layer_name, keep_mask, original_out, verbose)

            if verbose:
                print(f"  {layer_name}: {keep_mask.sum().item()} neurons remaining")

    def _adjust_batchnorm(self, conv_layer_name: str, keep_mask: torch.Tensor, verbose: bool = True) -> None:
        """Adjust BatchNorm after pruning Conv layer"""
        layers = list(self.model.named_modules())
        current_idx = next(i for i, (name, _) in enumerate(layers) if name == conv_layer_name)
        
        for i in range(current_idx + 1, min(current_idx + 3, len(layers))):
            next_name, next_layer = layers[i]
            
            if isinstance(next_layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if next_layer.weight is not None:
                    next_layer.weight = nn.Parameter(next_layer.weight.data[keep_mask])
                if next_layer.bias is not None:
                    next_layer.bias = nn.Parameter(next_layer.bias.data[keep_mask])
                
                if next_layer.running_mean is not None:
                    next_layer.running_mean = next_layer.running_mean[keep_mask]
                if next_layer.running_var is not None:
                    next_layer.running_var = next_layer.running_var[keep_mask]
                
                next_layer.num_features = keep_mask.sum().item()
                
                if verbose:
                    print(f"  Adjusted {next_name} (BatchNorm): {next_layer.num_features} features")
                
                break

    def _adjust_next_layers(self, current_layer_name: str,
                           keep_mask: torch.Tensor,
                           original_out_size: int,
                           verbose: bool = True) -> None:

        layers_dict = dict(self.model.named_modules())
        all_layers = list(self.model.named_modules())
        current_layer = layers_dict[current_layer_name]
        
        # Find all conv/linear layers that might consume this layer's output
        adjusted_layers = set()
        
        # Method 1: Sequential path (next layer in forward order)
        self._adjust_sequential_next(current_layer_name, keep_mask, original_out_size, 
                                     all_layers, adjusted_layers, verbose)
        
        # Method 2: Parallel paths (skip connections in ResNet)
        # Look for layers at the same hierarchical level that might be skip connections
        self._adjust_skip_connections(current_layer_name, keep_mask, original_out_size,
                                      layers_dict, adjusted_layers, verbose)

    def _adjust_sequential_next(self, current_layer_name: str,
                                keep_mask: torch.Tensor,
                                original_out_size: int,
                                all_layers: list,
                                adjusted_layers: set,
                                verbose: bool = True) -> None:
        """Adjust the next layer in sequential order."""
        current_idx = next(i for i, (name, _) in enumerate(all_layers) if name == current_layer_name)
        current_layer = all_layers[current_idx][1]

        for i in range(current_idx + 1, len(all_layers)):
            next_name, next_layer = all_layers[i]

            # Skip non-computational layers
            if isinstance(next_layer, (nn.BatchNorm2d, nn.BatchNorm1d, nn.ReLU, nn.ReLU6, 
                                       nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid,
                                       nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
                                       nn.Dropout, nn.Dropout2d)):
                continue

            if isinstance(next_layer, nn.Linear):
                if next_name in adjusted_layers:
                    continue
                    
                if isinstance(current_layer, (nn.Conv2d, nn.Conv1d)):
                    if next_layer.in_features % original_out_size == 0:
                        spatial_size = next_layer.in_features // original_out_size
                        new_in_features = keep_mask.sum().item() * spatial_size
                        
                        old_weight = next_layer.weight.data
                        old_weight_reshaped = old_weight.view(
                            next_layer.out_features,
                            original_out_size,
                            spatial_size
                        )
                        new_weight = old_weight_reshaped[:, keep_mask, :].reshape(
                            next_layer.out_features, -1
                        )
                        next_layer.weight = nn.Parameter(new_weight)
                        next_layer.in_features = new_in_features
                        adjusted_layers.add(next_name)
                        
                        if verbose:
                            print(f"  Adjusted {next_name} input: {new_in_features}")
                        break
                else:
                    if next_layer.in_features == keep_mask.numel():
                        next_layer.weight = nn.Parameter(next_layer.weight.data[:, keep_mask])
                        next_layer.in_features = keep_mask.sum().item()
                        adjusted_layers.add(next_name)
                        if verbose:
                            print(f"  Adjusted {next_name} input: {next_layer.in_features}")
                        break

            elif isinstance(next_layer, (nn.Conv2d, nn.Conv1d)):
                if next_name in adjusted_layers:
                    continue
                    
                if next_layer.in_channels == keep_mask.numel():
                    next_layer.weight = nn.Parameter(next_layer.weight.data[:, keep_mask, ...])
                    next_layer.in_channels = keep_mask.sum().item()
                    adjusted_layers.add(next_name)
                    if verbose:
                        print(f"  Adjusted {next_name} input: {next_layer.in_channels}")
                    break

    def _adjust_skip_connections(self, current_layer_name: str,
                                 keep_mask: torch.Tensor,
                                 original_out_size: int,
                                 layers_dict: dict,
                                 adjusted_layers: set,
                                 verbose: bool = True) -> None:
        """Adjust skip/shortcut connections in ResNet-style architectures"""
        # Parse layer name to understand hierarchy
        parts = current_layer_name.split('.')
        
        # Check if this is a layer that might have a skip connection at the next stage
        if len(parts) >= 3 and parts[0].startswith('layer') and parts[-1] in ['conv2', 'bn2']:
            try:
                layer_num = int(parts[0].replace('layer', ''))
                next_layer_name = f"layer{layer_num + 1}.0.downsample.0"
                
                if next_layer_name in layers_dict and next_layer_name not in adjusted_layers:
                    downsample_conv = layers_dict[next_layer_name]
                    
                    if isinstance(downsample_conv, (nn.Conv2d, nn.Conv1d)):
                        # This downsample takes the SAME input as the next layer's conv1
                        # So it needs the same input channel adjustment
                        if downsample_conv.in_channels == keep_mask.numel():
                            downsample_conv.weight = nn.Parameter(
                                downsample_conv.weight.data[:, keep_mask, ...]
                            )
                            downsample_conv.in_channels = keep_mask.sum().item()
                            adjusted_layers.add(next_layer_name)
                            
                            if verbose:
                                print(f"  ✓ Adjusted {next_layer_name} input (skip): {downsample_conv.in_channels}")
            except (ValueError, IndexError, KeyError):
                pass  # Not a ResNet-style layer, skip
