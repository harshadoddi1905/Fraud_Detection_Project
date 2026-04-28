# Fraud Detection using Anomaly Detection

## Project Overview
This project focuses on detecting fraudulent transactions using machine learning.
An **unsupervised anomaly detection model (Isolation Forest)** is used to identify suspicious transactions based on patterns in the data.



## Tools & Technologies
* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* VS Code



## Project Structure

```
Fraud_Detection_Project/
│
├── data/
│   └── transactions.csv
├── scripts/
│   └── fraud_detection.py
├── output/
│   └── fraud_results.csv
├── dashboard/
│   ├── fraud_distribution.png
│   └── risk_score.png
├── README.md
```


## Key Features
* Data cleaning and preprocessing
* Feature selection and scaling
* Anomaly detection using Isolation Forest
* Risk scoring for transactions
* Model evaluation using classification metrics
* Visualization of fraud patterns
* Identification of high-risk transactions



## Model Used
* **Isolation Forest (Unsupervised Learning)**
  * Detects anomalies based on data distribution
  * Works well for imbalanced datasets



## Model Performance
* Evaluated using:
  * Accuracy
  * Precision
  * Recall
  * Confusion Matrix
Fraud detection is an imbalanced problem, so recall is more important than accuracy.



## Visualizations
### 🔹 Fraud Distribution
![Fraud Distribution](dashboard/fraud_distribution.png)
### 🔹 Risk Score Distribution
![Risk Score](dashboard/risk_score.png)



## Key Insights
* High-value transactions are more likely to be fraudulent
* Transactions with high velocity (frequency) show higher risk
* Only a small percentage of transactions are fraud (imbalanced data)
* Risk score helps prioritize suspicious transactions



## How to Run
1. Clone the repository:
```
git clone https://github.com/yourusername/fraud-detection-project.git
```

2. Navigate to the folder:
```
cd Fraud_Detection_Project
```

3. Install dependencies:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the script:
```
python scripts/fraud_detection.py
```



## Output
* `fraud_results.csv` → Contains predictions and risk scores
* Dashboard images → Visual analysis



## Future Improvements
* Use advanced models (Random Forest, XGBoost)
* Hyperparameter tuning
* Real-time fraud detection system
* Dashboard using Power BI or Streamlit



##  Contact
Feel free to reach out for any questions or collaboration opportunities.