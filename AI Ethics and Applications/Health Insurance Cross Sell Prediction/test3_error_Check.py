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
    
    # Separate the gender column
    gender = data['Gender']
    
    # Drop columns that are not needed for model training
    X = pd.get_dummies(data.drop(columns=['Response', 'Gender']), drop_first=True)
    y = data['Response']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the feature values
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
def evaluate_model(model, X_test, y_test, gender):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and AUC
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Print the classification report
    class_report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{class_report}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot Confusion Matrix
    plot_confusion_matrix(cm, 'Overall Confusion Matrix')
    
    # Filter by gender
    male_indices = gender == 'Male'
    female_indices = gender == 'Female'
    
    # Separate confusion matrices for Male and Female
    cm_male = confusion_matrix(y_test[male_indices], y_pred[male_indices])
    cm_female = confusion_matrix(y_test[female_indices], y_pred[female_indices])
    
    plot_confusion_matrix(cm_male, 'Confusion Matrix for Male')
    plot_confusion_matrix(cm_female, 'Confusion Matrix for Female')

    return accuracy, auc, cm_male, cm_female, cm

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
# Step 5: Fairness Evaluation
def evaluate_fairness(cm_male, cm_female):
    # Calculate and print accuracy for Male and Female
    accuracy_male = (cm_male[0, 0] + cm_male[1, 1]) / cm_male.sum()
    accuracy_female = (cm_female[0, 0] + cm_female[1, 1]) / cm_female.sum()
    print(f"Accuracy for Male: {accuracy_male:.2f}")
    print(f"Accuracy for Female: {accuracy_female:.2f}")
    
    # Calculate Demographic Parity and Equal Opportunity
    positive_male = cm_male[1, 1] / cm_male.sum()
    positive_female = cm_female[1, 1] / cm_female.sum()
    print(f"Proportion of Positive Predictions for Male: {positive_male:.2f}")
    print(f"Proportion of Positive Predictions for Female: {positive_female:.2f}")
    
    tpr_male = cm_male[1, 1] / (cm_male[1, 1] + cm_male[1, 0])
    tpr_female = cm_female[1, 1] / (cm_female[1, 1] + cm_female[1, 0])
    print(f"True Positive Rate for Male: {tpr_male:.2f}")
    print(f"True Positive Rate for Female: {tpr_female:.2f}")

# %% 
# Main Flow
def main():
    # Load and preprocess the data
    file_path = "D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Data_Creation.xlsx"
    X_train_scaled, X_test_scaled, y_train, y_test, gender = load_and_prepare_data(file_path)
    
    # Train the model
    svm_model = train_svm(X_train_scaled, y_train)
    
    # Evaluate the model
    accuracy, auc, cm_male, cm_female, cm = evaluate_model(svm_model, X_test_scaled, y_test, gender)
    
    # Print results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC: {auc:.2f}")
    
    # Evaluate fairness
    evaluate_fairness(cm_male, cm_female)

# %% 
# Run the main function
if __name__ == "__main__":
    main()
