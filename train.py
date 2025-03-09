import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Goal::  Predicting Pod Lifetime

# Load the data
data = pd.read_csv('kubernetes_performance_metrics_dataset.csv')

data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")  # Convert to datetime
data["timestamp"] = data["timestamp"].astype("int64") // 10**9  # Convert to Unix timestamp (seconds)
data = data.drop(["pod_name", "namespace", "event_type", "event_message", "scaling_event"], axis=1)

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




# Split the dta into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Train an Isolation Forest model to detect outliers
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

y_random_forest_train = rf.predict(X_train)
y_random_forest_test = rf.predict(X_test)

# Should print the number of trees (default is 100)
print(rf.n_estimators)

