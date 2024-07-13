import random
import numpy as np
import torch
from scipy.signal import resample, butter, lfilter

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def resample_signal(signal, orig_freq, new_freq):
    """
    Resample signal from original frequency to new frequency.
    """
    num_samples = round(len(signal) * new_freq / orig_freq)
    resampled_signal = resample(signal, num_samples)
    return resampled_signal

def filter_signal(signal, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to the signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def scale_signal(signal):
    """
    Scale signal to have zero mean and unit variance.
    """
    scaled_signal = (signal - np.mean(signal)) / np.std(signal)
    return scaled_signal

def baseline_correction(signal, baseline_window):
    """
    Apply baseline correction.
    """
    baseline = np.mean(signal[:baseline_window])
    corrected_signal = signal - baseline
    return corrected_signal