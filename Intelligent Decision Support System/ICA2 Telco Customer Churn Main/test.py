# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
data = pd.read_csv('D:/Teesside University Semester 1/Inelligent Decision  System/ICA2 Group Work/Telco-Customer-Churn-main/Telco-Customer-Churn-main/telco.csv')  # Make sure the file path is correct

# Step 2: Drop unnecessary columns
data = data.drop(['Unnamed: 0', 'customerID'], axis=1)

# Step 3: Handle missing values
# Convert 'TotalCharges' to numeric, forcing non-numeric values to NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
# Fill missing values (NaNs) with the mean of the column
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
# Confirm no missing values remain
print(data.isnull().sum())  # To check if there are still any NaN values

# Step 4: Encode categorical columns as numbers (Ensure all categorical columns are encoded)
label_encoder = LabelEncoder()

# List of columns to encode (update this list based on your dataset)
categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PaymentMethod', 'InternetService']

# Encode each categorical column
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Step 5: Split data into features (X) and target (y)
X = data.drop('Churn', axis=1)  # Features (all columns except 'Churn')
y = data['Churn']  # Target ('Churn' column)

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Step 7: Train a Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy of the model:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Step 10: Save the trained model for later use
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)
