# %%
# Step No: 1 - Import Necessary Libraries
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# %%
# Step No: 2 - Load and Preprocess the Data
def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the dataset.
    
    Args:
    file_path (str): The file path to the dataset
    
    Returns:
    X_train_scaled, X_test_scaled, y_train, y_test, data: Processed training and testing data, and original data
    """
    # Load the dataset
    data = pd.read_excel(file_path)
    
    # Feature and target variables
    X = pd.get_dummies(data.drop(columns=['Response']), drop_first=True)
    y = data['Response']
    
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, data

# %%
# Step No: 3 - Train the SVM Model
def train_svm_model(X_train, y_train):
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

# %%
# Step No: 4 - Evaluate Model Performance
def evaluate_model(svm_model, X_test, y_test, data):
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])
    class_report = classification_report(y_test, y_pred)
    
    # Confusion Matrix for overall
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot the overall confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Overall Confusion Matrix")
    plt.colorbar()
    
    # Labels for the axes
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No', 'Yes'], rotation=45, fontsize=12)
    plt.yticks(tick_marks, ['No', 'Yes'], fontsize=12)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    
    # Annotate each cell with the numeric value
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=16,  # Increase font size for better readability
                     color="white" if cm[i, j] > thresh else "black")
    
    # Display the plot for overall confusion matrix
    plt.tight_layout()
    plt.show()
    
    # Segment the data by gender (assuming 'Gender' column exists in your data)
    male_indices = data['Gender'] == 'Male'  # Replace 'Gender' with the actual column name
    female_indices = data['Gender'] == 'Female'  # Replace 'Gender' with the actual column name

    # Male Data and Predictions
    X_test_male = X_test[male_indices]
    y_test_male = y_test[male_indices]
    y_pred_male = y_pred[male_indices]
    
    # Confusion Matrix for Male
    cm_male = confusion_matrix(y_test_male, y_pred_male)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm_male, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Male")
    plt.colorbar()
    
    # Annotate the confusion matrix for Male
    for i in range(cm_male.shape[0]):
        for j in range(cm_male.shape[1]):
            plt.text(j, i, format(cm_male[i, j], 'd'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=16,  # Increase font size for better readability
                     color="white" if cm_male[i, j] > cm_male.max() / 2.0 else "black")
    
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
    
    # Confusion Matrix for Female
    cm_female = confusion_matrix(y_test_female, y_pred_female)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm_female, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Female")
    plt.colorbar()
    
    # Annotate the confusion matrix for Female
    for i in range(cm_female.shape[0]):
        for j in range(cm_female.shape[1]):
            plt.text(j, i, format(cm_female[i, j], 'd'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=16,  # Increase font size for better readability
                     color="white" if cm_female[i, j] > cm_female.max() / 2.0 else "black")
    
    plt.xticks(tick_marks, ['No', 'Yes'], rotation=45, fontsize=12)
    plt.yticks(tick_marks, ['No', 'Yes'], fontsize=12)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return accuracy, auc, class_report


# %%
# Step No: 5 - Visualize the LIME Explanation
def visualize_lime_explanation(exp):
    """
    Visualizes the LIME explanation using a bar chart to show feature importance.
    """
    exp.as_pyplot_figure()
    plt.title("LIME Explanation")
    plt.show()

# %%
# Step No: 6 - Visualize the SHAP Summary Plot
def visualize_shap_summary(shap_values, X_test):
    """
    Visualizes the SHAP summary plot for the model.
    """
    shap.summary_plot(shap_values[1], X_test)
    plt.title("SHAP Summary Plot")
    plt.show()

# %%
# Step No: 7 - Explain the Model Using LIME
def explain_with_lime(X_train, y_train, X_test, instance_index=0):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, training_labels=y_train, mode='classification')
    exp = explainer.explain_instance(X_test[instance_index], svm_model.predict_proba)
    
    # Visualize the explanation
    visualize_lime_explanation(exp)
    
    return exp

# %%
# Step No: 8 - Main Execution Flow
def main():
    # Load and preprocess data
    file_path = "D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Data_Creation.xlsx"
    X_train_scaled, X_test_scaled, y_train, y_test, data = load_and_preprocess_data(file_path)
    
    # Train the SVM model
    svm_model = train_svm_model(X_train_scaled, y_train)
    
    # Evaluate model performance (Pass data argument here)
    accuracy, auc, class_report = evaluate_model(svm_model, X_test_scaled, y_test, data)
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    print(f"Classification Report:\n{class_report}")
    
    # Explain model predictions using LIME
    print("\nExplaining with LIME (for instance 0):")
    exp = explain_with_lime(X_train_scaled, y_train, X_test_scaled, instance_index=0)
    
    # Explain model predictions using SHAP
    print("\nExplaining with SHAP:")
    shap_values = explain_with_shap(X_train_scaled, X_test_scaled, svm_model)

# %%
# Step 9: Execute the main function
if __name__ == "__main__":
    main()

# %%
