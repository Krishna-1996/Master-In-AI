# %% 
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Machine (SVM) for binary classification
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# %% 
# Step 2: Load and Preprocess Dataset
df = pd.read_csv('D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Health Insurance Main Dataset - Copy.csv')

# Encode categorical variables
label_encoder = LabelEncoder()

# Encode 'Gender' (Male = 0, Female = 1)
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Encode 'Vehicle_Age' (Convert '> 2 Years', '1-2 Year', '< 1 Year' to numerical values)
df['Vehicle_Age'] = df['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2})

# Encode 'Vehicle_Damage' (Yes = 1, No = 0)
df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})

# Normalize numerical features
scaler = StandardScaler()
df[['Age', 'Annual_Premium', 'Vintage']] = scaler.fit_transform(df[['Age', 'Annual_Premium', 'Vintage']])

# %% 
# Step 3: Split Data into Features (X) and Target (y)
X = df.drop(columns=['id', 'Response'])  # Drop 'id' and 'Response' columns
y = df['Response']  # 'Response' is the target variable (0 or 1)

# Split the data (70% for training, 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% 
# Step 4: Initialize and Train Support Vector Machine (SVM) Model
model = SVC(kernel='linear', class_weight='balanced', random_state=42)  # Linear kernel for simplicity
model.fit(X_train, y_train)

# %% 
# Step 5: Make Predictions and Evaluate Performance
y_pred = model.predict(X_test)

# Evaluate the model with various metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.decision_function(X_test))  # Use decision function for SVM AUC

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# %% 
# Step 6: Visualizations
# 1. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.decision_function(X_test))
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# %% 
# Step 7: Fairness Evaluation Based on Gender
y_pred_test = model.predict(X_test)

# Create a DataFrame to hold the results, including 'Gender'
results = X_test.copy()  # Copy the features, as they have 'Gender' already included
results['actual'] = y_test
results['predicted'] = y_pred_test

# Split results by gender
male_results = results[results['Gender'] == 0]  # Male = 0
female_results = results[results['Gender'] == 1]  # Female = 1

# %% 
# Step 8: Fairness Metrics Calculations
# Accuracy for each group
male_accuracy = accuracy_score(male_results['actual'], male_results['predicted'])
female_accuracy = accuracy_score(female_results['actual'], female_results['predicted'])

# Demographic Parity (Proportion of positive predictions for each group)
male_positive_rate = male_results['predicted'].mean()
female_positive_rate = female_results['predicted'].mean()

# Equal Opportunity (True Positive Rate for each group)
male_tpr = recall_score(male_results['actual'], male_results['predicted'])
female_tpr = recall_score(female_results['actual'], female_results['predicted'])

# Output fairness metrics
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
# Step 9: Create a comparison table for performance and fairness metrics
comparison_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'ROC AUC', 'Male Accuracy', 'Female Accuracy', 
               'Male Positive Rate (Demographic Parity)', 'Female Positive Rate (Demographic Parity)', 
               'Male True Positive Rate (Equal Opportunity)', 'Female True Positive Rate (Equal Opportunity)'],
    'Value': [accuracy, recall, precision, f1, roc_auc, male_accuracy, female_accuracy, 
              male_positive_rate, female_positive_rate, male_tpr, female_tpr]
})

print(comparison_table)

# %% 
# Step 10: Check Imbalance for Categorical Features
categorical_features = ['Gender', 'Vehicle_Age', 'Vehicle_Damage', 'Response']  # Add more categorical features if needed

# Print the class distribution for each categorical feature
for feature in categorical_features:
    print(f"Class Distribution for {feature}:")
    print(df[feature].value_counts(normalize=True))  # Normalize to get percentage
    print("\n")

# %%
# Step 11: Visualize Distribution for Numerical Features
numerical_features = ['Age', 'Annual_Premium', 'Vintage']  # Add more numerical features if needed

# Plot histograms for numerical features to check for balance
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
