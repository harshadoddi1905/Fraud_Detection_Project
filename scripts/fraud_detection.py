import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

print("FRAUD DETECTION PROJECT")


# Create folders
os.makedirs("output", exist_ok=True)
os.makedirs("dashboard", exist_ok=True)

# Load data
df = pd.read_csv("data/transactions.csv")

print("\nData Preview:")
print(df.head())

# DATA CLEANING
df = df.dropna()

print("\nData Shape:", df.shape)



features = [
    'amount',
    'transaction_hour',
    'velocity_last_24h',
    'cardholder_age'
]
X = df[features]

# SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# MODEL 
model = IsolationForest(contamination=0.02, random_state=42)
df['anomaly'] = model.fit_predict(X_scaled)

# Convert to fraud label
df['predicted_fraud'] = df['anomaly'].map({-1:1, 1:0})

# RISK SCORE 
df['risk_score'] = model.decision_function(X_scaled)

# Lower score = more risky
df['risk_score'] = -df['risk_score']

# EVALUATION 
print("\nMODEL PERFORMANCE:")

print("\nClassification Report:")
print(classification_report(df['is_fraud'], df['predicted_fraud']))

cm = confusion_matrix(df['is_fraud'], df['predicted_fraud'])
print("\nConfusion Matrix:")
print(cm)

# VISUALIZATION

# Fraud distribution
plt.figure(figsize=(5,4))
df['predicted_fraud'].value_counts().plot(kind='bar', color=['blue','red'])
plt.title("Predicted Fraud Distribution")
plt.savefig("dashboard/fraud_distribution.png")
plt.close()

# Risk score distribution
plt.figure(figsize=(6,4))
sns.histplot(df['risk_score'], bins=50, kde=True)
plt.title("Risk Score Distribution")
plt.savefig("dashboard/risk_score.png")
plt.close()

# TOP FRAUD TRANSACTIONS
top_fraud = df.sort_values(by='risk_score', ascending=False).head(10)

print("\nTOP 10 HIGH-RISK TRANSACTIONS:")
print(top_fraud[['transaction_id','amount','risk_score']])

# SAVE OUTPUT
df.to_csv("output/fraud_results.csv", index=False)