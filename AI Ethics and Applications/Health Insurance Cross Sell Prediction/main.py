# %%
Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score, roc_auc_score, roc_curve

# %%
Step 2: Load and Preprocess Dataset
# Load dataset (replace with your dataset)
df = pd.read_csv('D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Health Insurance Main Dataset.csv')

print(df.head(5))

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
Step 3: Split Data into Features (X) and Target (y)
# Split into features (X) and target (y)
X = df.drop(columns=['id', 'Response'])  # Drop 'id' and 'Response' columns
y = df['Response']  # 'Response' is the target variable (0 or 1)

# Split the data (70% for training, 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
Step 4: Initialize and Train Logistic Regression Model
# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# %%
Step 5: Make Predictions and Evaluate Performance
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model with various metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

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
Step 6: Visualizations

# 1. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 3. Confusion Matrix for Each Feature
# Plot confusion matrix for each feature with the target feature
features = ['Age', 'Annual_Premium', 'Vintage', 'Vehicle_Age', 'Vehicle_Damage', 'Gender']
for feature in features:
    feature_matrix = confusion_matrix(y_test, model.predict(X_test[feature].values.reshape(-1, 1)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(feature_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f'Confusion Matrix for {feature}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# %%
Step 7: Fairness Evaluation Based on Gender
# Get predicted values for the test set
y_pred_test = model.predict(X_test)

# Create a DataFrame to hold the results, including 'Gender'
results = X_test.copy()  # Copy the features, as they have 'Gender' already included
results['actual'] = y_test
results['predicted'] = y_pred_test

# Split results by gender
male_results = results[results['Gender'] == 0]  # Male = 0
female_results = results[results['Gender'] == 1]  # Female = 1

# %%
Step 8: Evaluate Fairness Metrics
# Calculate accuracy for each group
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
Step 9: Tabular Comparison of Metrics
# Create a comparison table for performance and fairness metrics
comparison_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'ROC AUC', 'Male Accuracy', 'Female Accuracy', 
               'Male Positive Rate (Demographic Parity)', 'Female Positive Rate (Demographic Parity)', 
               'Male True Positive Rate (Equal Opportunity)', 'Female True Positive Rate (Equal Opportunity)'],
    'Value': [accuracy, recall, precision, f1, roc_auc, male_accuracy, female_accuracy, 
              male_positive_rate, female_positive_rate, male_tpr, female_tpr]
})

print(comparison_table)

# %%
Step 10: Discuss Limitations and Trade-offs
# Provide analysis on the limitations of fairness metrics
print("""
Fairness Evaluation Discussion:

1. **Accuracy:** While accuracy is a useful metric, it can be misleading in imbalanced datasets, as the model could have high accuracy by simply predicting the majority class. This does not necessarily indicate fairness.

2. **Demographic Parity (Positive Rate):** A disparity in positive prediction rates between genders might indicate bias toward one group. However, this metric doesn't account for potential differences in base rates between groups.

3. **Equal Opportunity (True Positive Rate):** This metric evaluates whether both groups have equal chances of being correctly classified as positive. However, achieving equal opportunity does not guarantee overall fairness, as trade-offs may exist with other metrics.

4. **ROC AUC:** The ROC AUC score provides a better measure of the model's ability to distinguish between classes across various thresholds, providing a more comprehensive evaluation beyond just accuracy.

In real-world applications, there's often a trade-off between these fairness metrics. For instance, improving accuracy may negatively affect demographic parity or equal opportunity, and vice versa. A careful balance should be sought depending on the specific fairness objectives of the application.
""")
