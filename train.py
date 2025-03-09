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
data = data.drop(["pod_name"], axis=1)
from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()
data["event_type"] = label_encoder.fit_transform(data["event_type"])
data["namespace"] = label_encoder.fit_transform(data["namespace"])
data["scaling_event"] = data["scaling_event"].astype(int)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=100)  # Limit features to avoid high dimensionality
event_message_tfidf = vectorizer.fit_transform(data['event_message'].fillna(""))

event_message_df = pd.DataFrame(event_message_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# Drop original event_message column and merge new TF-IDF features
data = data.drop(columns=['event_message']).reset_index(drop=True)
data = pd.concat([data, event_message_df], axis=1)

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




# Split the dta into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train an Isolation Forest model to detect outliers
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=4, random_state=50)
rf.fit(X_train, y_train)


y_random_forest_train = rf.predict(X_train)
y_random_forest_test = rf.predict(X_test)

# Should print the number of trees (default is 100)
print(rf.n_estimators)


print(f"Training R² score: {rf.score(X_train, y_train):.4f}")
print(f"Test R² score: {rf.score(X_test, y_test):.4f}")


y_pred = rf.predict(X_test[:5])
print("Predicted:", y_pred)
print("Actual:", y_test[:5].values)




