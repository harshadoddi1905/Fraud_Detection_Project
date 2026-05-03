# Fraud Detection using Machine Learning

# Project Overview

This project focuses on detecting fraudulent transactions using both:

* Isolation Forest (unsupervised anomaly detection)
* Logistic Regression (supervised classification)

The dataset is highly imbalanced (~1.5% fraud), so the focus is on optimizing precision and recall instead of accuracy.


# Approach

* Data preprocessing and feature scaling (StandardScaler)
* Anomaly detection using Isolation Forest
* Probability-based classification using Logistic Regression
* Threshold tuning to balance fraud detection vs false positives


## Final Results (Logistic Regression - Threshold = 0.8)

* Precision (Fraud): **0.13**
* Recall (Fraud): **0.40**
* F1 Score: **0.19**



```
Confusion Matrix:
[[9435 414]
 [ 91  60]]
```


# Key Insight

Lower thresholds increased fraud detection but caused too many false positives.
Threshold = 0.8 provided the best trade-off for practical use.


# Tools & Technologies
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
|   └── roc_curve.png  
├── README.md
```


##  How to Run

1. Clone the repository:

```bash
git clone https://github.com/harshadoddi1905/Fraud_Detection_Project.git
```

2. Navigate to the folder:

```bash
cd Fraud_Detection_Project
```

3. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the script:

```bash
python scripts/fraud_detection.py
```

---




## Key Features
* Data cleaning and preprocessing
* Feature selection and scaling
* Anomaly detection using Isolation Forest
* Risk scoring for transactions
* Model evaluation using classification metrics
* Visualization of fraud patterns
* Identification of high-risk transactions



## Models Used

### 1. Isolation Forest (Unsupervised Learning)
* Detects anomalies based on data distribution
* Works well for imbalanced datasets
* Used to generate risk scores for transactions

### 2. Logistic Regression (Supervised Learning)
* Used for binary classification (fraud vs non-fraud)
* Applied with class balancing to handle imbalanced data
* Threshold tuning used to improve recall



## Model Performance
* Evaluated using:
  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * Confusion Matrix
  * ROC-AUC Score
Fraud detection is an imbalanced problem, so recall is more important than accuracy.



## Visualizations
###  Fraud Distribution
![Fraud Distribution](dashboard/fraud_distribution.png)
###  Risk Score Distribution
![Risk Score](dashboard/risk_score.png)
### ROC Curve
![ROC Curve](dashboard/roc_curve.png)



## Key Insights
* Fraud detection is a highly imbalanced problem (~0.17% fraud)
* Recall is more important than accuracy in fraud detection
* Lower threshold increases fraud detection but raises false positives
* Logistic Regression with threshold tuning improved recall significantly
* ROC-AUC score (~0.74) indicates reasonable model performance





## Output
* fraud_results.csv` → Contains transaction predictions and risk scores
* Dashboard images → Visualizations of fraud patterns and risk distribution



## Future Improvements
* Implement advanced models (Random Forest, XGBoost)
* Perform hyperparameter tuning for better accuracy
* Build real-time fraud detection system
* Develop interactive dashboard using Streamlit or Power BI



##  Contact
Feel free to reach out for any questions or collaboration opportunities.
