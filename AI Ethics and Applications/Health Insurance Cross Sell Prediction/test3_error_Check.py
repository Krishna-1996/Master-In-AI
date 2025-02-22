# %% 
# Step No: 1 - Import Necessary Libraries
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import os

# %% 
# Step No: 2 - Load and Preprocess the Data
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    # Load the dataset
    data = pd.read_excel(file_path)
    
    # Check if the 'Response' column exists
    if 'Response' not in data.columns:
        raise ValueError("'Response' column not found in the dataset.")
    
    X = pd.get_dummies(data.drop(columns=['Response']), drop_first=True)
    y = data['Response']
    
    # Check if 'Gender' column exists
    if 'Gender' not in X.columns:
        raise ValueError("'Gender' column not found in the dataset.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, X, data

# %% 
# Step No: 3 - Train the SVM Model
def train_svm_model(X_train, y_train):
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

# %% 
# Step No: 4 - Evaluate Model Performance
def evaluate_model(svm_model, X_test, y_test, data, X):
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])
    class_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix for overall
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
    
    # Segment the data by gender using the original X DataFrame (before scaling)
    male_indices = X.loc[X_test.index, 'Gender'] == 'Male'
    female_indices = X.loc[X_test.index, 'Gender'] == 'Female'

    # Male Data and Predictions
    X_test_male = X_test[male_indices]
    y_test_male = y_test[male_indices]
    y_pred_male = y_pred[male_indices]
    
    cm_male = confusion_matrix(y_test_male, y_pred_male)
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

    return accuracy, auc, class_report, cm_male, cm_female, cm

# %% 
# Step No: 5 - Fairness Evaluation
def fairness_metrics(cm_male, cm_female, cm):
    # Calculate Equal Accuracy
    accuracy_male = (cm_male[1, 1] + cm_male[0, 0]) / cm_male.sum()
    accuracy_female = (cm_female[1, 1] + cm_female[0, 0]) / cm_female.sum()
    print(f"Accuracy for Male: {accuracy_male}")
    print(f"Accuracy for Female: {accuracy_female}")
    
    # Calculate Demographic Parity (Proportion of positive predictions)
    positive_male = cm_male[1, 1] / cm_male.sum()
    positive_female = cm_female[1, 1] / cm_female.sum()
    print(f"Proportion of positive predictions for Male: {positive_male}")
    print(f"Proportion of positive predictions for Female: {positive_female}")
    
    # Calculate Equal Opportunity (True Positive Rate)
    tpr_male = cm_male[1, 1] / (cm_male[1, 1] + cm_male[1, 0])
    tpr_female = cm_female[1, 1] / (cm_female[1, 1] + cm_female[1, 0])
    print(f"True Positive Rate for Male: {tpr_male}")
    print(f"True Positive Rate for Female: {tpr_female}")

# %% 
# Step No: 6 - Visualize Comparison and Insights
def visualize_comparisons(cm_male, cm_female, cm):
    # Bar plot for confusion matrix comparison
    cm_values = {'Male': cm_male.ravel(), 'Female': cm_female.ravel(), 'Overall': cm.ravel()}
    cm_df = pd.DataFrame(cm_values, columns=['TP', 'FP', 'FN', 'TN'])
    cm_df.plot(kind='bar', figsize=(10, 6))
    plt.title("Confusion Matrix Comparison (Male vs Female vs Overall)")
    plt.xticks([0], ['Confusion Matrix Values'], rotation=0)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Print the confusion matrix results in tabular form
    print(f"\nConfusion Matrix Results:\n{'='*30}")
    print(f"Overall Confusion Matrix:\n{cm}")
    print(f"Male Confusion Matrix:\n{cm_male}")
    print(f"Female Confusion Matrix:\n{cm_female}")

# %% 
# Step No: 7 - Main Execution Flow
def main():
    # Load and preprocess data
    file_path = "D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Data_Creation.xlsx"
    X_train_scaled, X_test_scaled, y_train, y_test, X, data = load_and_preprocess_data(file_path)
    
    # Train the SVM model
    svm_model = train_svm_model(X_train_scaled, y_train)
    
    # Evaluate model performance
    accuracy, auc, class_report, cm_male, cm_female, cm = evaluate_model(svm_model, X_test_scaled, y_test, data, X)
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    print(f"Classification Report:\n{class_report}")
    
    # Fairness Evaluation
    fairness_metrics(cm_male, cm_female, cm)
    
    # Visualize comparison and confusion matrix results
    visualize_comparisons(cm_male, cm_female, cm)

# %% 
# Execute the main function
if __name__ == "__main__":
    main()
