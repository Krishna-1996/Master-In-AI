# %% 
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# %% 
# Step 2: Load and Prepare the Data
def load_and_prepare_data(file_path):
    # Load the data
    data = pd.read_excel(file_path)
    
    # Separate the gender column and the response column
    gender = data['Gender']
    X = pd.get_dummies(data.drop(columns=['Response', 'Gender']), drop_first=True)
    y = data['Response']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, gender

# %% 
# Step 3: Train the SVM Model
def train_svm(X_train, y_train):
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

# %% 
# Step 4: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and AUC score
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Print the classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, 'Overall Confusion Matrix')

    return accuracy, auc, cm, y_pred

# %% 
# Plot Confusion Matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()

# %% 
# Step 5: Fairness Metrics (Equal Opportunity and Demographic Parity)
def fairness_metrics(cm_male, cm_female):
    # Calculate Equal Accuracy
    accuracy_male = (cm_male[1, 1] + cm_male[0, 0]) / cm_male.sum()
    accuracy_female = (cm_female[1, 1] + cm_female[0, 0]) / cm_female.sum()
    print(f"Accuracy for Male: {accuracy_male:.2f}")
    print(f"Accuracy for Female: {accuracy_female:.2f}")
    
    # Calculate Demographic Parity (Proportion of positive predictions)
    positive_male = cm_male[1, 1] / cm_male.sum()
    positive_female = cm_female[1, 1] / cm_female.sum()
    print(f"Proportion of positive predictions for Male: {positive_male:.2f}")
    print(f"Proportion of positive predictions for Female: {positive_female:.2f}")
    
    # Calculate Equal Opportunity (True Positive Rate)
    tpr_male = cm_male[1, 1] / (cm_male[1, 1] + cm_male[1, 0])
    tpr_female = cm_female[1, 1] / (cm_female[1, 1] + cm_female[1, 0])
    print(f"True Positive Rate for Male: {tpr_male:.2f}")
    print(f"True Positive Rate for Female: {tpr_female:.2f}")

# %% 
# Visualize Comparison of Confusion Matrices (Male vs Female vs Overall)
def visualize_comparisons(cm_male, cm_female, cm):
    cm_values = {
        'Male': cm_male.ravel(),
        'Female': cm_female.ravel(),
        'Overall': cm.ravel()
    }
    cm_df = pd.DataFrame(cm_values, columns=['TP', 'FP', 'FN', 'TN'])
    cm_df.plot(kind='bar', figsize=(10, 6))
    plt.title("Confusion Matrix Comparison (Male vs Female vs Overall)")
    plt.xticks([0], ['Confusion Matrix Values'], rotation=0)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"\nConfusion Matrix Results:\n{'='*30}")
    print(f"Overall Confusion Matrix:\n{cm}")
    print(f"Male Confusion Matrix:\n{cm_male}")
    print(f"Female Confusion Matrix:\n{cm_female}")

# %% 
# Step 6: Main Execution Flow
def main():
    # Load and prepare data
    file_path = "D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Data_Creation.xlsx"
    X_train_scaled, X_test_scaled, y_train, y_test, gender = load_and_prepare_data(file_path)
    
    # Train the SVM model
    svm_model = train_svm(X_train_scaled, y_train)
    
    # Evaluate the model
    accuracy, auc, cm, y_pred = evaluate_model(svm_model, X_test_scaled, y_test)

    # Split the data based on gender (without any errors)
    male_indices = (gender[X_test.index] == 'Male')
    female_indices = (gender[X_test.index] == 'Female')
    
    # Create confusion matrices for male and female
    cm_male = confusion_matrix(y_test[male_indices], y_pred[male_indices])
    cm_female = confusion_matrix(y_test[female_indices], y_pred[female_indices])

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC: {auc:.2f}")
    
    # Fairness Evaluation
    fairness_metrics(cm_male, cm_female)
    
    # Visualize confusion matrices comparison
    visualize_comparisons(cm_male, cm_female, cm)

# %% 
# Run the main function
if __name__ == "__main__":
    main()

# %%
