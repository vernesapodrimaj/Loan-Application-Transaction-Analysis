# Loan Application & Transaction Fraud Analysis

## Dataset

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
- Export cleaned data for dashboarding

---

## Tools Used

- Python (`pandas`, `seaborn`, `matplotlib`)
- Jupyter Notebook
- GitHub (version control & documentation)
- Tableau 

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

# Explore & clean
df_loan.info()
df_loan.describe()
df_loan.isnull().sum()

# Drop columns with >95% missing
df_loan = df_loan.dropna(thresh=len(df_loan)*0.05, axis=1)

# Convert to datetime
df_loan['application_date'] = pd.to_datetime(df_loan['application_date'], errors='coerce')
df_loan['application_month'] = df_loan['application_date'].dt.month
df_loan['application_dayofweek'] = df_loan['application_date'].dt.day_name()

# Categorical checks
df_loan['loan_type'].value_counts()
df_loan['purpose_of_loan'].value_counts()
df_loan['employment_status'].value_counts()

# Age bins
bins_age = [20, 30, 40, 50, 60, 70]
labels_age = ['21-30', '31-40', '41-50', '51-60', '61-65']
df_loan['age_group'] = pd.cut(df_loan['applicant_age'], bins=bins_age, labels=labels_age)

# Loan amount bins
bins_loan = [100000, 300000, 600000, 900000, 1200000, 1500000, 1700000]
labels_loan = ['100k-300k', '300k-600k', '600k-900k', '900k-1.2M', '1.2M-1.5M', '1.5M+']
df_loan['loan_amount_group'] = pd.cut(df_loan['loan_amount_requested'], bins=bins_loan, labels=labels_loan, include_lowest=True)

highest_loans = df_loan.groupby('customer_id')['loan_amount_requested'].max().sort_values(ascending=False)
print(highest_loans.head(10))

# Basic checks for Transaction
df_transaction.info()
df_transaction.describe()
df_transaction.isnull().sum()

# Convert to datetime
df_transaction['transaction_date'] = pd.to_datetime(df_transaction['transaction_date'], errors='coerce')
df_transaction['month'] = df_transaction['transaction_date'].dt.month
df_transaction['day_of_week'] = df_transaction['transaction_date'].dt.day_name()

# Value counts
df_transaction['transaction_type'].value_counts()
df_transaction['merchant_category'].value_counts()
df_transaction['transaction_location'].value_counts()

# Total amount by month
df_transaction.groupby('month')['transaction_amount'].sum().plot(kind='bar', title='Total Transaction Amount by Month')
plt.tight_layout()
plt.show()

# Count by transaction type per month
pd.crosstab(df_transaction['month'], df_transaction['transaction_type']).plot(kind='bar', stacked=True)
plt.title('Transaction Types by Month')
plt.tight_layout()
plt.show()

# Merchant category totals
df_transaction.groupby('merchant_category')['transaction_amount'].sum().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Top Merchant Categories by Transaction Amount')
plt.tight_layout()
plt.show()

# Total amount by month
df_transaction.groupby('month')['transaction_amount'].sum().plot(kind='bar', title='Total Transaction Amount by Month')
plt.tight_layout()
plt.show()

# Count by transaction type per month
pd.crosstab(df_transaction['month'], df_transaction['transaction_type']).plot(kind='bar', stacked=True)
plt.title('Transaction Types by Month')
plt.tight_layout()
plt.show()

# Merchant category totals
df_transaction.groupby('merchant_category')['transaction_amount'].sum().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Top Merchant Categories by Transaction Amount')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df_loan, x='loan_type', hue='employment_status', palette='Set2')
plt.title('Loan Type by Employment Status')
plt.xlabel('Loan Type')
plt.ylabel('Number of Applications')
plt.xticks(rotation=45)
plt.legend(title='Employment Status')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_loan, x='age_group', y='loan_amount_requested', palette='Set3')
plt.title('Loan Amount Requested by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Loan Amount Requested')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.countplot(data=df_loan, x='application_dayofweek', order=order, palette='pastel')
plt.title('Loan Applications by Day of the Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Applications')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df_transaction, x='month', hue='transaction_type', palette='Set1')
plt.title('Transaction Type by Month')
plt.xlabel('Month')
plt.ylabel('Transaction Count')
plt.legend(title='Transaction Type')
plt.tight_layout()
plt.show()

df_merged.to_csv("cleaned_loan_transaction_data.csv", index=False)



