# %%
# Step 0: Import Necessary Libraries
# df = pd.read_excel('D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Data_Creation.xlsx')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# Load dataset
data = pd.read_excel("D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Data_Creation.xlsx")

# %%
# Step 1: Preprocessing
# Convert categorical features to numeric
label_encoder = LabelEncoder()

# Encode Gender (Female = 0, Male = 1)
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Encode other categorical features (e.g., Vehicle_Age, Vehicle_Damage)
data['Vehicle_Age'] = label_encoder.fit_transform(data['Vehicle_Age'])
data['Vehicle_Damage'] = label_encoder.fit_transform(data['Vehicle_Damage'])

# %%
# Step 2: Prepare features and target
X = data.drop(columns=['Response'])
y = data['Response']

# %%
# Step 3: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Step 4: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# %%
# Step 5: Predictions
y_pred = model.predict(X_test)

# %%
# Step 6: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Accuracy: {accuracy}")
print(f"AUC: {roc_auc}")
print(classification_report(y_test, y_pred))

# %%
# Step 7: Split predictions based on Gender
# Add predictions to the original dataset (test set)
X_test['Gender'] = data.loc[X_test.index, 'Gender']
X_test['Predicted_Response'] = y_pred
X_test['True_Response'] = y_test

# Evaluate fairness by splitting on Gender
grouped = X_test.groupby('Gender')

# For each gender group, calculate metrics (Accuracy, Demographic Parity, Equal Opportunity)
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

# Display fairness metrics for each gender
for gender, metrics in gender_metrics.items():
    print(f"Metrics for Gender {gender}:")
    print(f"Accuracy: {metrics['Accuracy']}")
    print(f"Demographic Parity: {metrics['Demographic Parity']}")
    print(f"True Positive Rate: {metrics['True Positive Rate']}")
    print()

# %%
# Step 8: Conclusion
# If there is a significant difference in these metrics between genders, the model may be biased.

# %%
