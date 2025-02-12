# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset (replace with your dataset)
df = pd.read_csv('health_insurance_cross_sell.csv')

# Check for missing values
df.isnull().sum()

# Handle missing values (for simplicity, we will drop rows with missing data)
df = df.dropna()

# Encode categorical features (e.g., 'sex', 'region code')
label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])

# Feature engineering (example: scale numerical features)
scaler = StandardScaler()
df[['age', 'annual_premium', 'policy_sales_channel']] = scaler.fit_transform(df[['age', 'annual_premium', 'policy_sales_channel']])

# Split dataset into features (X) and target variable (y)
X = df.drop(columns=['response'])  # Assuming 'response' is the target variable
y = df['response']



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%
