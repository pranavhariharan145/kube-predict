import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Goal::  Predicting Pod Lifetime

# Load the data
data = pd.read_csv('kubernetes_performance_metrics_dataset.csv')
print(data.head())


