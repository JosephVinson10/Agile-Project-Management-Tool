import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the log data
data = pd.read_csv('security_logs.csv')

# Parse timestamps
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Risk score calculation
def calculate_risk(row):
    severity_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    return severity_mapping[row['Severity']]

data['RiskScore'] = data.apply(calculate_risk, axis=1)

# Prepare data for anomaly detection
data['Day'] = data['Timestamp'].dt.day
data['Hour'] = data['Timestamp'].dt.hour

# Train Isolation Forest for anomaly detection
features = data[['RiskScore', 'Day', 'Hour']]
model = IsolationForest(contamination=0.1, random_state=42)
data['Anomaly'] = model.fit_predict(features)

# Label anomalies (-1 = anomaly, 1 = normal)
data['AnomalyLabel'] = data['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# Save processed data with anomalies to a new file
data.to_csv('processed_security_logs.csv', index=False)

print("Processed data with anomalies saved to 'processed_security_logs.csv'")
