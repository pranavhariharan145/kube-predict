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


# Data Seperation 
# Possible X values: timestamp (Possibly useful if you're doing time-series analysis)
# pod_name (Might be categorical)
# namespace (Might be categorical)
# cpu_allocation_efficiency
# memory_allocation_efficiency
# disk_io
# network_latency
# node_temperature
# node_cpu_usage
# node_memory_usage
# scaling_event (Might be categorical)

# Possible Y values: pod_lifetime_seconds

y = data['pod_lifetime_seconds']
X = data.drop(['pod_lifetime_seconds'], axis=1)

print(X.head())
print(y.head())

