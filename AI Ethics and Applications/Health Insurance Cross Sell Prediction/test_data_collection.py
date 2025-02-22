# %% 
# Step No: 1 - Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, recall_score
import seaborn as sns
# %% 
# Step No: 2 - Load and Preprocess the Data
file_path = "D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Data_Creation.xlsx"
data = pd.read_excel(file_path)

# Preprocess the data: Use One-Hot-Encoding (get_Dummies)
X = pd.get_dummies(data.drop(columns=['Response']), drop_first=True)
y = data['Response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the same scale for all feature 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% 
# Step No: 3 - Train the SVM Model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# %% 
# Step No: 4 - Evaluate Model Performance
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, svm_model.predict_proba(X_test_scaled)[:, 1])
class_report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"AUC: {auc}")
print(f"Classification Report:\n{class_report}")

# Plot the overall confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Overall Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No', 'Yes'], rotation=45, fontsize=12)
plt.yticks(tick_marks, ['No', 'Yes'], fontsize=12)
plt.xlabel('Predicted label', fontsize=14)
plt.ylabel('True label', fontsize=14)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 fontsize=16,  color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# %% 
# Step No: 5 - Segment the Data by Gender (assuming 'Gender' column exists)
male_indices = data.loc[X_test.index, 'Gender'] == 'Male'
female_indices = data.loc[X_test.index, 'Gender'] == 'Female'

# Male Data and Predictions
X_test_male = X_test[male_indices]
y_test_male = y_test[male_indices]
y_pred_male = y_pred[male_indices]

cm_male = confusion_matrix(y_test_male, y_pred_male)

# Plot Male Confusion Matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm_male, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Male")
plt.colorbar()

for i in range(cm_male.shape[0]):
    for j in range(cm_male.shape[1]):
        plt.text(j, i, format(cm_male[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 fontsize=16, color="white" if cm_male[i, j] > cm_male.max() / 2.0 else "black")

plt.xticks(tick_marks, ['No', 'Yes'], rotation=45, fontsize=12)
plt.yticks(tick_marks, ['No', 'Yes'], fontsize=12)
plt.xlabel('Predicted label', fontsize=14)
plt.ylabel('True label', fontsize=14)
plt.tight_layout()
plt.show()

# Female Data and Predictions
X_test_female = X_test[female_indices]
y_test_female = y_test[female_indices]
y_pred_female = y_pred[female_indices]

cm_female = confusion_matrix(y_test_female, y_pred_female)

# Plot Female Confusion Matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm_female, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Female")
plt.colorbar()

for i in range(cm_female.shape[0]):
    for j in range(cm_female.shape[1]):
        plt.text(j, i, format(cm_female[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 fontsize=16, color="white" if cm_female[i, j] > cm_female.max() / 2.0 else "black")

plt.xticks(tick_marks, ['No', 'Yes'], rotation=45, fontsize=12)
plt.yticks(tick_marks, ['No', 'Yes'], fontsize=12)
plt.xlabel('Predicted label', fontsize=14)
plt.ylabel('True label', fontsize=14)
plt.tight_layout()
plt.show()

# %% 
# Step No: 6 - Visualize Comparison and Insights
# Flatten the confusion matrices into arrays and organize them in a dictionary for plotting
cm_values = {
    'Confusion Matrix': ['TP', 'FP', 'FN', 'TN'],
    'Male': cm_male.ravel(),
    'Female': cm_female.ravel(),
    'Overall': cm.ravel()
}

# Create a DataFrame with confusion matrix values in a structured format
cm_df = pd.DataFrame(cm_values)

# Set 'Confusion Matrix' column as the index
cm_df.set_index('Confusion Matrix', inplace=True)

# Plot a bar graph for confusion matrix comparison
cm_df.plot(kind='bar', figsize=(10, 6))

plt.title("Confusion Matrix Comparison (Male vs Female vs Overall)")
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.show()

# Print the confusion matrix results in tabular form for reference
print(f"\nConfusion Matrix Results:\n{'='*30}")
print(f"Overall Confusion Matrix:\n{cm}")
print(f"Male Confusion Matrix:\n{cm_male}")
print(f"Female Confusion Matrix:\n{cm_female}")

# %%
# Step 7: Fairness Evaluation Based on Gender
# Equal Accuracy:
accuracy_male = accuracy_score(y_test_male, y_pred_male)
accuracy_female = accuracy_score(y_test_female, y_pred_female)
print(f"Accuracy for Male: {accuracy_male}")
print(f"Accuracy for Female: {accuracy_female}")

# Demographic Parity (Proportion of predicted positives):
prop_positive_male = np.sum(y_pred_male == 1) / len(y_pred_male)
prop_positive_female = np.sum(y_pred_female == 1) / len(y_pred_female)
print(f"Demographic Parity (Male): {prop_positive_male}")
print(f"Demographic Parity (Female): {prop_positive_female}")

# Equal Opportunity (Recall/True Positive Rate):
recall_male = recall_score(y_test_male, y_pred_male)
recall_female = recall_score(y_test_female, y_pred_female)
print(f"Recall for Male: {recall_male}")
print(f"Recall for Female: {recall_female}")

# %%
