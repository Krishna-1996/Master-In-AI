# %%
# Step 1: Load and Explore the Dataset

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_excel('D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Data_Creation.xlsx')

# Basic dataset exploration
print(df.head())  # Show first few rows
print(df.info())  # Check data types and missing values

# Handle missing values (simple strategy, can be adjusted based on the dataset)
df.fillna(df.mean(), inplace=True)

# Encode categorical variables to numeric if needed (for example, converting 'male'/'female' to 0/1)
df = pd.get_dummies(df)

# Assuming the target variable is 'target' and the protected characteristic is 'gender' or similar
# Ensure the target variable is binary (0/1)
df['target'] = df['target'].map({'yes': 1, 'no': 0})  # Replace with actual target column mapping if needed
# %%


# %%
# Step 2: Train a Basic SVM Model

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split the data into features and target variable
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target variable

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
# %%


# %%
# Step 3: Evaluate Bias Using Fairness Metrics

# Assuming the protected characteristic is 'gender' (binary)
# Split the data into groups based on the protected characteristic
group_1 = df[df['gender'] == 0]  # For example, males (0)
group_2 = df[df['gender'] == 1]  # For example, females (1)

# Predict outcomes for both groups
y_pred_group_1 = svm_model.predict(group_1.drop('target', axis=1))
y_pred_group_2 = svm_model.predict(group_2.drop('target', axis=1))

# Calculate accuracy for both groups
accuracy_group_1 = accuracy_score(group_1['target'], y_pred_group_1)
accuracy_group_2 = accuracy_score(group_2['target'], y_pred_group_2)

# Print out the accuracy per group
print(f"Accuracy for Group 1 (Gender=0): {accuracy_group_1}")
print(f"Accuracy for Group 2 (Gender=1): {accuracy_group_2}")

# Demographic Parity (positive prediction rate across groups)
positive_rate_group_1 = np.mean(y_pred_group_1)
positive_rate_group_2 = np.mean(y_pred_group_2)

print(f"Positive Prediction Rate for Group 1: {positive_rate_group_1}")
print(f"Positive Prediction Rate for Group 2: {positive_rate_group_2}")

# Equal Opportunity (True Positive Rate across groups)
true_positive_group_1 = np.sum((y_pred_group_1 == 1) & (group_1['target'] == 1)) / np.sum(group_1['target'] == 1)
true_positive_group_2 = np.sum((y_pred_group_2 == 1) & (group_2['target'] == 1)) / np.sum(group_2['target'] == 1)

print(f"True Positive Rate for Group 1: {true_positive_group_1}")
print(f"True Positive Rate for Group 2: {true_positive_group_2}")
# %%


# %%
# Step 4: Mitigate Bias Using LISM and SHAP

import shap

# Create a SHAP explainer object for the SVM model
explainer = shap.KernelExplainer(svm_model.predict_proba, X_train)

# Get SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Plot the SHAP summary plot to see the most important features
shap.summary_plot(shap_values, X_test)

# For LISM, you would need to implement or use a custom library for subgroup-based training
# Example: using sklearn's SVM but accounting for subgroup weights or using an alternative method for subgroup correction
# Here, I won't add full implementation for LISM since it might require more advanced handling or an additional library.
# %%


# %%
# Step 5: Compare the Results

# After applying LISM or SHAP, re-evaluate fairness metrics (the code for fairness metrics is already shown in Step 3)
# You can compare the results from the SHAP-based analysis with the initial results to see if bias is reduced.

# In case you modify the model after applying LISM or SHAP:
# retrain the model, check fairness metrics again, and compare changes to the initial metrics.

