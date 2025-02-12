# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset (replace with your dataset)
df = pd.read_csv('D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Health Insurance Main Dataset.csv')
# D:\Masters Projects\Master-In-AI\AI Ethics and Applications\Health Insurance Cross Sell Prediction
print(df.head(5))

# %%
# Check for missing values
df.isnull().sum()

# Handle missing values (for simplicity, we will drop rows with missing data)
df = df.dropna()

# Encode categorical features (e.g., 'sex', 'region code')
label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['Gender'])

# Feature engineering (example: scale numerical features)
scaler = StandardScaler()
df[['age', 'annual_premium', 'policy_sales_channel']] = scaler.fit_transform(df[['age', 'annual_premium', 'policy_sales_channel']])

# Split dataset into features (X) and target variable (y)
X = df.drop(columns=['response'])  # Assuming 'response' is the target variable
y = df['response']



# %%
# Split the dataset into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model using various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")



# %%
# Add the 'sex' column to the test set for fairness evaluation
X_test['sex'] = X_test['sex']

# Get predictions for the test set
y_pred_test = model.predict(X_test.drop(columns=['sex']))

# Create a DataFrame to hold the results
results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_test,
    'sex': X_test['sex']
})

# Separate the results by sex
male_results = results[results['sex'] == 0]  # Assuming 0 is male
female_results = results[results['sex'] == 1]  # Assuming 1 is female

# Calculate accuracy for each group
male_accuracy = accuracy_score(male_results['actual'], male_results['predicted'])
female_accuracy = accuracy_score(female_results['actual'], female_results['predicted'])

# Demographic Parity (Proportion of positive predictions for each group)
male_positive_rate = male_results['predicted'].mean()
female_positive_rate = female_results['predicted'].mean()

# Equal Opportunity (True Positive Rate for each group)
male_tpr = recall_score(male_results['actual'], male_results['predicted'])
female_tpr = recall_score(female_results['actual'], female_results['predicted'])

# Output the fairness metrics
print(f"Male Accuracy: {male_accuracy}")
print(f"Female Accuracy: {female_accuracy}")
print(f"Difference in Accuracy: {male_accuracy - female_accuracy}")

print(f"Male Positive Rate (Demographic Parity): {male_positive_rate}")
print(f"Female Positive Rate (Demographic Parity): {female_positive_rate}")
print(f"Difference in Demographic Parity: {male_positive_rate - female_positive_rate}")

print(f"Male True Positive Rate (Equal Opportunity): {male_tpr}")
print(f"Female True Positive Rate (Equal Opportunity): {female_tpr}")
print(f"Difference in True Positive Rate: {male_tpr - female_tpr}")



# %%



# %%



# %%



# %%



# %%



# %%
