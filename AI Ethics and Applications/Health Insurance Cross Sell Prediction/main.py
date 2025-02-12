
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score

# Load dataset (replace with your dataset)
df = pd.read_csv('D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Health Insurance Main Dataset.csv')
# D:\Masters Projects\Master-In-AI\AI Ethics and Applications\Health Insurance Cross Sell Prediction
print(df.head(5))
# Encode categorical variables
label_encoder = LabelEncoder()

# Encode 'Gender' (Male = 0, Female = 1)
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Encode 'Vehicle_Age' (Convert '> 2 Years', '1-2 Year', '< 1 Year' to numerical values)
df['Vehicle_Age'] = df['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2})

# Encode 'Vehicle_Damage' (Yes = 1, No = 0)
df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})

# Normalize numerical features
scaler = StandardScaler()
df[['Age', 'Annual_Premium', 'Vintage']] = scaler.fit_transform(df[['Age', 'Annual_Premium', 'Vintage']])

# Split into features (X) and target (y)
X = df.drop(columns=['id', 'Response'])  # Drop 'id' and 'Response' columns
y = df['Response']  # 'Response' is the target variable (0 or 1)

# Split the data (70% for training, 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model with various metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")

# Fairness Evaluation based on Gender

# Get predicted values for the test set (no need to add 'Gender' to X_test again)
y_pred_test = model.predict(X_test)

# Create a DataFrame to hold the results, including 'Gender'
results = X_test.copy()  # Copy the features, as they have 'Gender' already included
results['actual'] = y_test
results['predicted'] = y_pred_test

# Split results by gender
male_results = results[results['Gender'] == 0]  # Male = 0
female_results = results[results['Gender'] == 1]  # Female = 1

# Calculate accuracy for each group
male_accuracy = accuracy_score(male_results['actual'], male_results['predicted'])
female_accuracy = accuracy_score(female_results['actual'], female_results['predicted'])

# Demographic Parity (Proportion of positive predictions for each group)
male_positive_rate = male_results['predicted'].mean()
female_positive_rate = female_results['predicted'].mean()

# Equal Opportunity (True Positive Rate for each group)
male_tpr = recall_score(male_results['actual'], male_results['predicted'])
female_tpr = recall_score(female_results['actual'], female_results['predicted'])

# Output fairness metrics
print(f"Male Accuracy: {male_accuracy}")
print(f"Female Accuracy: {female_accuracy}")
print(f"Difference in Accuracy: {male_accuracy - female_accuracy}")

print(f"Male Positive Rate (Demographic Parity): {male_positive_rate}")
print(f"Female Positive Rate (Demographic Parity): {female_positive_rate}")
print(f"Difference in Demographic Parity: {male_positive_rate - female_positive_rate}")

print(f"Male True Positive Rate (Equal Opportunity): {male_tpr}")
print(f"Female True Positive Rate (Equal Opportunity): {female_tpr}")
print(f"Difference in True Positive Rate: {male_tpr - female_tpr}")