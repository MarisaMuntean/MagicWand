import tensorflow as tf
import pandas as pd
import math

# Hyperparameter settings (these numbers aren't arbitrary!)
timesteps = 200      # Time steps = sampling rate(100Hz) × time(2 seconds)
lr = 1e-4           # Learning rate = 0.0001, too big causes instability, too small learns slowly
num_epochs = 200    # Training rounds, like practicing basketball shots repeatedly
batch_size = 4      # Batch size, learning from 4 samples at once
input_dim = 2       # Input dimensions: Y-axis and Z-axis acceleration
num_classes = 4     # 4 gestures: rock(0), scissors(1), paper(2), no action(3)