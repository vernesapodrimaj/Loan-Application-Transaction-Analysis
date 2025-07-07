# 💳 Loan Application & Transaction Fraud Analysis

## 📁 Dataset

**Source:** [Kaggle - Loan Application and Transaction Fraud Detection](https://www.kaggle.com/datasets/prajwaldongre/loan-application-and-transaction-fraud-detection)

This project analyzes loan applications and financial transactions to uncover patterns related to loan requests, fraud risk, customer behavior, and transaction characteristics. The dataset includes:

- `loan_applications.csv`: applicant-level info (age, loan type, income, fraud flag, etc.)
- `transactions.csv`: transaction-level data (amount, type, merchant category, etc.)

---

## Objectives

- Clean and explore both datasets (applications & transactions)
- Extract time-based patterns (month, weekday)
- Bin loan amounts and applicant ages into readable groups
- Identify high-risk behaviors and customer profiles
- Merge datasets to explore deeper patterns and correlations
- Export cleaned data for modeling or dashboarding

---

## Tools Used

- Python (`pandas`, `seaborn`, `matplotlib`)
- Jupyter Notebook
- GitHub (version control & documentation)
- Tableau (optional dashboarding)

---

## Data Cleaning & Feature Engineering

### Load Data

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV files
df_loan = pd.read_csv('/Users/vernesapodrimaj/Downloads/archive-2/loan_applications.csv')
df_transaction = pd.read_csv('/Users/vernesapodrimaj/Downloads/archive-2/transactions.csv')

