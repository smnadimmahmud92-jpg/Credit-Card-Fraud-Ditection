Credit Card Fraud Detection | Machine Learning Project

A complete end-to-end **supervised machine learning classification project** for detecting fraudulent credit card transactions using a **Random Forest Classifier**. The project demonstrates real-world handling of **highly imbalanced financial transaction data**, feature importance analysis, and probabilistic risk scoring.

---

 Problem Statement

Credit card fraud causes **billions of dollars in losses annually**. The goal of this project is to develop a machine learning system that can:

- Accurately classify fraudulent vs legitimate transactions  
- Handle **severe class imbalance**
- Provide **risk probabilities** instead of only class labels  

---

Dataset Description

- Source: Kaggle — European cardholders transaction dataset  
- Observations: **284,807 transactions**  
- Fraud Cases: **492 (~0.17%)**  
- Features:  
  - `V1`–`V28` (PCA-transformed, anonymized)
  - `Time`
  - `Amount`
  - `Class` (Target Variable)

> ⚠️ Dataset is **not uploaded** due to size and privacy restrictions.  
> 🔗 Download: https://www.kaggle.com/mlg-ulb/creditcardfraud

---

## Tech Stack

- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  
- **Model:** Random Forest Classifier  
- **Environment:** Google Colab / Jupyter Notebook  

---

##  Methodology

1. **Data Loading & Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Class Imbalance Handling**
4. **Train-Test Split**
5. **Model Training using Random Forest**
6. **Feature Importance Analysis**
7. **Model Evaluation Metrics**
8. **Custom Risk-Score Prediction Function**

---

##  Machine Learning Model

- Algorithm: **Random Forest Classifier**
- Objective: Binary Classification (Fraud = 1, Normal = 0)
- Output: Fraud Probability Score

### Example Prediction Function:

```python
predict_transaction(transaction_data)
