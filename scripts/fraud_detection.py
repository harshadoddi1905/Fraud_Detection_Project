import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression

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

# Logistic Regression Model
lr_model = LogisticRegression(class_weight='balanced')
lr_model.fit(X_scaled, df['is_fraud'])

df['lr_prob'] = lr_model.predict_proba(X_scaled)[:,1]

df['lr_pred'] = (df['lr_prob'] > 0.8).astype(int)


# MODEL 
model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(X_scaled)



# RISK SCORE 
df['risk_score'] = model.decision_function(X_scaled)

# Lower score = more risky
df['risk_score'] = -df['risk_score']


# Normalize risk score (0 to 1)
df['risk_score'] = (df['risk_score'] - df['risk_score'].min()) / (df['risk_score'].max() - df['risk_score'].min())

# Set threshold (you can tune this)
threshold = 0.7

# New fraud prediction based on risk score
df['predicted_fraud'] = (df['risk_score'] > threshold).astype(int)
# EVALUATION 
print("\nMODEL PERFORMANCE:")

#print("\nClassification Report:")
#print(classification_report(df['is_fraud'], df['predicted_fraud']))
print("\nISOLATION FOREST PERFORMANCE:")
print(classification_report(df['is_fraud'], df['predicted_fraud']))

print("\nLOGISTIC REGRESSION PERFORMANCE:")
print(classification_report(df['is_fraud'], df['lr_pred']))

cm = confusion_matrix(df['is_fraud'], df['lr_pred'])
print("\nConfusion Matrix:")
print(cm)

# ROC Curve
fpr, tpr, _ = roc_curve(df['is_fraud'], df['risk_score'])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("dashboard/roc_curve.png")
plt.close()

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