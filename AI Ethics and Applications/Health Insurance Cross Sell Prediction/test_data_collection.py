# %%
#Step 1: Import necessary libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
# Step 2: Load dataset
data = pd.read_excel("D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Data_Creation.xlsx")

# %%
# Step 3: Preprocess and split the dataset
X = data.drop(columns=['Response'])
y = data['Response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Step 4: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Step 5: Train the SVM model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# %%
# Step 6: Make predictions
y_pred = svm_model.predict(X_test_scaled)

# %%
# Step 7: Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test_scaled)[:, 1])

print(f"Accuracy: {accuracy}")
print(f"AUC: {roc_auc}")
print(classification_report(y_test, y_pred))

# %%
# Step 8: Fairness Evaluation based on Gender
X_test['Gender'] = data.loc[X_test.index, 'Gender']
X_test['Predicted_Response'] = y_pred
X_test['True_Response'] = y_test

grouped = X_test.groupby('Gender')
gender_metrics = {}

for gender, group in grouped:
    group_accuracy = accuracy_score(group['True_Response'], group['Predicted_Response'])
    demographic_parity = group['Predicted_Response'].mean()
    cm = confusion_matrix(group['True_Response'], group['Predicted_Response'])
    true_positive_rate = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) != 0 else 0
    gender_metrics[gender] = {
        "Accuracy": group_accuracy,
        "Demographic Parity": demographic_parity,
        "True Positive Rate": true_positive_rate
    }

for gender, metrics in gender_metrics.items():
    print(f"Metrics for Gender {gender}:")
    print(f"Accuracy: {metrics['Accuracy']}")
    print(f"Demographic Parity: {metrics['Demographic Parity']}")
    print(f"True Positive Rate: {metrics['True Positive Rate']}")
    print()

# %%
# Step 9: Using LIME for explanation (local interpretability)
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_scaled, training_labels=y_train, mode='classification')
i = 10  # Pick any instance from test set
exp = explainer.explain_instance(X_test_scaled[i], svm_model.predict_proba)
exp.show_in_notebook()  # This shows the explanation for the chosen instance
