"""
Script untuk generate sample EEG data untuk testing Streamlit app
Jalankan: python generate_sample_data.py
Output: data/sample_training.csv dan data/sample_online.csv
"""

import numpy as np
import pandas as pd
import os

# Konfigurasi
SAMPLING_RATE = 200  # Hz
DURATION = 20  # detik
N_CHANNELS = 20

# Create output folder
os.makedirs('data', exist_ok=True)

# Generate realistic EEG-like signal
def generate_eeg_signal(duration, fs, n_channels, seed=42):
    """
    Generate realistic synthetic EEG signal
    
    Args:
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        n_channels: Number of EEG channels
        seed: Random seed for reproducibility
    
    Returns:
        numpy array of shape (n_samples, n_channels)
    """
    n_samples = int(duration * fs)
    time = np.linspace(0, duration, n_samples)
    
    signal = np.zeros((n_samples, n_channels))
    
    for ch in range(n_channels):
        # Seed untuk variasi antar channel
        np.random.seed(seed + ch)
        
        # Delta (0.5-4 Hz)
        delta = 2.0 * np.sin(2 * np.pi * 2.0 * time + np.random.rand())
        
        # Theta (4-8 Hz)
        theta = 1.5 * np.sin(2 * np.pi * 6.0 * time + np.random.rand())
        
        # Alpha (8-13 Hz)
        alpha = 3.0 * np.sin(2 * np.pi * 10.0 * time + np.random.rand())
        
        # Beta (13-30 Hz)
        beta = 1.0 * np.sin(2 * np.pi * 20.0 * time + np.random.rand())
        
        # Gamma (30-45 Hz)
        gamma = 0.5 * np.sin(2 * np.pi * 40.0 * time + np.random.rand())
        
        # Combine frequencies
        signal[:, ch] = delta + theta + alpha + beta + gamma
        
        # Add noise
        noise = np.random.normal(0, 0.5, n_samples)
        signal[:, ch] += noise
        
        # Add slight powerline interference (50 Hz)
        powerline = 0.3 * np.sin(2 * np.pi * 50.0 * time)
        signal[:, ch] += powerline
    
    return signal


# Generate TRAINING data
print("="*80)
print("GENERATING SAMPLE EEG DATA")
print("="*80 + "\n")

print("1. Generating TRAINING data...")
training_data = generate_eeg_signal(DURATION, SAMPLING_RATE, N_CHANNELS, seed=42)

print(f"   Shape: {training_data.shape}")
print(f"   Duration: {DURATION} seconds")
print(f"   Sampling rate: {SAMPLING_RATE} Hz")
print(f"   Channels: {N_CHANNELS}")

# Save TRAINING (NO HEADER, pure numeric)
training_file = 'data/sample_training.csv'
pd.DataFrame(training_data).to_csv(training_file, index=False, header=False)

print(f"   ✓ Saved: {training_file}")
print(f"   File size: {os.path.getsize(training_file) / 1024:.2f} KB\n")

# Generate ONLINE data (different seed = different pattern)
print("2. Generating ONLINE data...")
online_data = generate_eeg_signal(DURATION, SAMPLING_RATE, N_CHANNELS, seed=123)

print(f"   Shape: {online_data.shape}")
print(f"   Duration: {DURATION} seconds")
print(f"   Sampling rate: {SAMPLING_RATE} Hz")
print(f"   Channels: {N_CHANNELS}")

# Save ONLINE (NO HEADER, pure numeric)
online_file = 'data/sample_online.csv'
pd.DataFrame(online_data).to_csv(online_file, index=False, header=False)

print(f"   ✓ Saved: {online_file}")
print(f"   File size: {os.path.getsize(online_file) / 1024:.2f} KB\n")

# Statistics
print("="*80)
print("DATA STATISTICS")
print("="*80)

print("\nTRAINING Data:")
print(f"  Mean: {training_data.mean():.4f}")
print(f"  Std : {training_data.std():.4f}")
print(f"  Min : {training_data.min():.4f}")
print(f"  Max : {training_data.max():.4f}")

print("\nONLINE Data:")
print(f"  Mean: {online_data.mean():.4f}")
print(f"  Std : {online_data.std():.4f}")
print(f"  Min : {online_data.min():.4f}")
print(f"  Max : {online_data.max():.4f}")

print("\n" + "="*80)
print("✅ DONE!")
print("="*80)
print("\nYou can now test the Streamlit app with:")
print(f"  • {training_file}")
print(f"  • {online_file}")
print("\nBoth files are in CSV format without headers (pure numeric data)")
print("Format: (samples, channels) = (4000, 20)")
print("="*80 + "\n")