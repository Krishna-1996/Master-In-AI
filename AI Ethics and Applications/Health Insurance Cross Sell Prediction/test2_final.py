# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular
import shap
import numpy as np

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the dataset.
    
    Args:
    file_path (str): The file path to the dataset
    
    Returns:
    X_train_scaled, X_test_scaled, y_train, y_test: Processed training and testing data
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

# Function to train the SVM model
def train_svm_model(X_train, y_train):
    """
    Trains the SVM model using the provided training data.
    
    Args:
    X_train (array): Feature matrix for training
    y_train (array): Target labels for training
    
    Returns:
    svm_model (SVC): Trained SVM model
    """
    # Train an SVM classifier
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    return svm_model

# Function to evaluate model performance
def evaluate_model(svm_model, X_test, y_test):
    """
    Evaluates the performance of the trained model using various metrics.
    
    Args:
    svm_model (SVC): Trained SVM model
    X_test (array): Feature matrix for testing
    y_test (array): True labels for testing
    
    Returns:
    dict: Dictionary containing accuracy, AUC, and classification report
    """
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])
    class_report = classification_report(y_test, y_pred)
    
    return {
        "accuracy": accuracy,
        "AUC": auc,
        "classification_report": class_report
    }

# Function to evaluate fairness metrics
def evaluate_fairness(X_test, y_test, y_pred, protected_attribute='Gender'):
    """
    Evaluates the fairness of the model based on the selected protected attribute.
    
    Args:
    X_test (DataFrame): Test feature matrix
    y_test (array): True labels for testing
    y_pred (array): Predicted labels
    protected_attribute (str): The protected attribute for fairness evaluation (default is 'Gender')
    
    Returns:
    dict: Dictionary containing fairness metrics for each group (e.g., Gender)
    """
    # Adding gender and prediction data to the test set
    X_test['Gender'] = X_test['Gender']  # Ensure 'Gender' column exists in X_test
    X_test['Predicted_Response'] = y_pred
    X_test['True_Response'] = y_test
    
    # Grouping data by protected attribute (Gender)
    grouped = X_test.groupby(protected_attribute)
    
    # Initialize dictionary to store fairness metrics
    fairness_metrics = {}
    
    # Calculate fairness metrics for each group
    for group_name, group in grouped:
        group_accuracy = accuracy_score(group['True_Response'], group['Predicted_Response'])
        demographic_parity = group['Predicted_Response'].mean()
        cm = confusion_matrix(group['True_Response'], group['Predicted_Response'])
        true_positive_rate = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) != 0 else 0
        
        fairness_metrics[group_name] = {
            "Accuracy": group_accuracy,
            "Demographic Parity": demographic_parity,
            "True Positive Rate": true_positive_rate
        }
    
    return fairness_metrics

# Function to explain the model using LIME
def explain_with_lime(X_train, y_train, X_test, instance_index=0):
    """
    Uses LIME to explain the model's prediction for a given instance.
    
    Args:
    X_train (array): Training feature matrix
    y_train (array): Training labels
    X_test (array): Test feature matrix
    instance_index (int): The index of the instance to explain (default is 0)
    
    Returns:
    exp: LIME explanation object
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, training_labels=y_train, mode='classification')
    exp = explainer.explain_instance(X_test[instance_index], svm_model.predict_proba)
    
    return exp

# Function to explain the model using SHAP
def explain_with_shap(X_train, X_test, svm_model):
    """
    Uses SHAP to explain the model's predictions.
    
    Args:
    X_train (array): Training feature matrix
    X_test (array): Test feature matrix
    svm_model (SVC): Trained SVM model
    
    Returns:
    shap_values: SHAP values for the test set
    """
    explainer = shap.KernelExplainer(svm_model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test)
    
    # Plot SHAP summary plot
    shap.summary_plot(shap_values[1], X_test)  # Plot for class '1'
    
    return shap_values

# Main Execution Flow
def main():
    # Load and preprocess data
    file_path = "D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Data_Creation.xlsx"
    X_train_scaled, X_test_scaled, y_train, y_test, data = load_and_preprocess_data(file_path)
    
    # Train the SVM model
    svm_model = train_svm_model(X_train_scaled, y_train)
    
    # Evaluate model performance
    evaluation_results = evaluate_model(svm_model, X_test_scaled, y_test)
    print("Model Evaluation Results:")
    print(f"Accuracy: {evaluation_results['accuracy']}")
    print(f"AUC: {evaluation_results['AUC']}")
    print(f"Classification Report:\n{evaluation_results['classification_report']}")
    
    # Evaluate fairness metrics
    y_pred = svm_model.predict(X_test_scaled)
    fairness_metrics = evaluate_fairness(X_test, y_test, y_pred)
    print("\nFairness Evaluation Results:")
    for group, metrics in fairness_metrics.items():
        print(f"\nMetrics for {group}:")
        print(f"Accuracy: {metrics['Accuracy']}")
        print(f"Demographic Parity: {metrics['Demographic Parity']}")
        print(f"True Positive Rate: {metrics['True Positive Rate']}")
    
    # Explain model predictions using LIME
    print("\nExplaining with LIME (for instance 0):")
    exp = explain_with_lime(X_train_scaled, y_train, X_test_scaled, instance_index=0)
    exp.show_in_notebook()  # Shows the explanation in the notebook
    
    # Explain model predictions using SHAP
    print("\nExplaining with SHAP:")
    shap_values = explain_with_shap(X_train_scaled, X_test_scaled, svm_model)

# Execute the main function
if __name__ == "__main__":
    main()
