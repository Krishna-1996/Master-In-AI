# %% 
# 1.1 Load the Dataset (Dynamically fetch file in same folder as .py script)
import os
import pandas as pd

# Get the current directory where the .py file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'Student Level Prediction Using Machine Learning - Copy.csv')

# Load the dataset
df = pd.read_csv(file_path)

# %% 
# 2. Preprocess the Data (Assumed)
# Assuming you have preprocessed the data as follows:
# - X is the feature set (independent variables)
# - y is the target variable (dependent variable)

X = df.drop(columns='Target')  # Replace 'Target' with your actual target column name
y = df['Target']  # Replace 'Target' with your actual target column name

# %% 
# 3. Split the Data into Training and Test Sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% 
# 4. Initialize the Models
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier

# Define models
models = {
    'SVM': SVC(probability=True, random_state=42),
    'Voting Classifier': VotingClassifier(estimators=[('rf', RandomForestClassifier()), ('svm', SVC(probability=True))], voting='soft'),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# %% 
# 5. Hyperparameter Tuning (Example with GridSearchCV)
from sklearn.model_selection import GridSearchCV

# Define hyperparameters for tuning (example for Random Forest)
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}

# Example GridSearch for Random Forest
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Get the best model and parameters
best_rf_model = grid_search_rf.best_estimator_
print(f"Best Random Forest Model: {best_rf_model}")
print(f"Best Hyperparameters: {grid_search_rf.best_params_}")

# Update the models dictionary to include the best tuned model
models['Random Forest'] = best_rf_model

# %% 
# 6. Train the Models
for model_name, model in models.items():
    model.fit(X_train, y_train)

# %% 
# 7. Evaluate the Models on the Test Set
from sklearn.metrics import accuracy_score

model_accuracies = {}
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[model_name] = accuracy

print("Model Accuracies:", model_accuracies)

# %% 
# 8. Plot ROC Curves for Each Model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Plot ROC curve for each model
for model_name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc='lower right')
        plt.show()

# %% 
# 9. Generate Predictions & Save Results to CSV
# Combine actual values and predictions for each model
data = X_test.copy()
data['Actual_Value'] = y_test

for model_name, model in models.items():
    data[f'Predict_Value_{model_name}'] = model.predict(X_test)

# Save predictions to CSV
output_path = os.path.join(current_dir, 'predictions_output.csv')
data.to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")

# %% 
# 10. LIME for Model Interpretability (Explaining Instance Predictions)
import lime.lime_tabular

# Initialize LIME explainers for each model
explainers = {model_name: lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns,
    class_names=['0', '1'],
    mode='classification'
) for model_name in models}

# User input to select test instance for explanation
index_to_check = int(input("Enter the index of the instance to explain: ")) - 1

# Ensure the index is valid
if 0 <= index_to_check < len(X_test):
    instance = X_test.iloc[index_to_check]
    actual_value = y_test.iloc[index_to_check]

    # Get model predictions for the instance
    predictions = {model_name: model.predict(instance.values.reshape(1, -1))[0] for model_name, model in models.items()}

    print(f"\nInstance {index_to_check + 1} - Actual Value: {actual_value}")
    for model_name, prediction in predictions.items():
        print(f"{model_name} Predicted Value: {prediction}")
    
    # Create subplots for LIME explanations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (model_name, explainer) in zip(axes.flat, explainers.items()):
        explanation = explainer.explain_instance(instance.values, models[model_name].predict_proba, num_features=10)
        explanation.as_pyplot_figure(label=1).axes[0].set_title(f'{model_name} Explanation')
    
    plt.tight_layout()
    plt.show()

else:
    print("Invalid index. Please enter a valid index from the test data.")

# %% 
# 11. Save LIME Feature Importances to CSV
def get_lime_feature_importances(explanation, model_name):
    feature_importances = explanation.as_list()
    sorted_feature_importances = sorted(feature_importances, key=lambda x: abs(x[1]), reverse=True)
    return [{'Model': model_name, 'Feature': feature, 'Importance': importance} for feature, importance in sorted_feature_importances]

# Collect and save all feature importances
all_feature_importances = []
if 0 <= index_to_check < len(X_test):
    instance = X_test.iloc[index_to_check]
    for model_name, explainer in explainers.items():
        explanation = explainer.explain_instance(instance.values, models[model_name].predict_proba, num_features=10)
        feature_importances = get_lime_feature_importances(explanation, model_name)
        all_feature_importances.extend(feature_importances)

    # Save the feature importances to a CSV file
    feature_importance_df = pd.DataFrame(all_feature_importances)
    output_csv_path = os.path.join(current_dir, 'lime_feature_importances.csv')
    feature_importance_df.to_csv(output_csv_path, index=False)
    print(f"Feature importances saved to: {output_csv_path}")
else:
    print("Invalid index. Please enter a valid index from the test data.")
