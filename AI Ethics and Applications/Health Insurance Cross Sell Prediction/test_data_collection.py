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

# %%
# Step No: 3 - Train the SVM Model
def train_svm_model(X_train, y_train):
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

# %%
# Step No: 4 - Evaluate Model Performance
def evaluate_model(svm_model, X_test, y_test):
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])
    class_report = classification_report(y_test, y_pred)
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No', 'Yes'], rotation=45)
    plt.yticks(tick_marks, ['No', 'Yes'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
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
    
    # Evaluate model performance
    accuracy, auc, class_report = evaluate_model(svm_model, X_test_scaled, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    print(f"Classification Report:\n{class_report}")
    
    # Explain model predictions using LIME
    print("\nExplaining with LIME (for instance 0):")
    exp = explain_with_lime(X_train_scaled, y_train, X_test_scaled, instance_index=0)
    
    
# %%
# Execute the main function
if __name__ == "__main__":
    main()

# %%
