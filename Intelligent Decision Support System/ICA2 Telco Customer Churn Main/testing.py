# %%
# Import Necessary Libraries
import numpy as np
import pandas as pd
import pandas_profiling
from scipy.stats import mode
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %%
# Import Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# %%
# Import train test split
from sklearn.model_selection import train_test_split

# %%
# Import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# %%
# Pickle to save model
import pickle

# %%
# Visualization libraries
import seaborn as sns

# %%
# Manage warnings
import warnings
warnings.filterwarnings('ignore')

# %%
# Loading Data
data = pd.read_csv('D:/Teesside University Semester 1/Inelligent Decision System/ICA2 Group Work/Telco-Customer-Churn-main/Telco-Customer-Churn-main/telco.csv')

# %%
# Dropping Unnecessary Columns
data = data.drop(['Unnamed: 0', 'customerID'], axis=1)

# %%
# Fixing NULL values and replacing with mean
data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
data.fillna(data.mean(), inplace=True)
data['TotalCharges'] = data['TotalCharges'].astype('float64')
data.fillna(data.mean(), inplace=True)

# %%
# Checking for Null Values
data.nunique()
print(data.isnull().sum())

# %%
# Outlier detection on continuous variables
outliers = data[['MonthlyCharges', 'TotalCharges']]
def out():
    outliers = data[['MonthlyCharges', 'TotalCharges']]
    for i in outliers:
        sns.boxplot(data=outliers)

out()

# %%
# Plotting Pie Chart for Contract Type
import plotly.express as px

fig = px.pie(data['Contract'], values="Contract", names="Contract") 
fig.show()

# %%
# Plotting Bar Chart for Payment Method
fig = px.bar(data['PaymentMethod'],
             x='PaymentMethod',
             y='PaymentMethod',
             title='Test',
             color='PaymentMethod',
             barmode='stack')

fig.show()

# %%
# Plotting Density Plot for Tenure
import seaborn as sns

# Make default density plot
sns.kdeplot(data['tenure'])

# %%
# KPI-1
df1 = pd.DataFrame(data, columns=['tenure'])

av_column = df1.mean(axis=0)
print(av_column)
print('------')

import plotly.graph_objects as go

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=32.37,
    title={'text': "Average"},
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig.show()

# %%
# KPI-2
df2 = pd.DataFrame(data, columns=['MonthlyCharges'])

av_column = df2.mean(axis=0)
print(av_column)

fig = go.Figure(go.Indicator(
    mode="number+gauge+delta",
    gauge={'shape': "bullet"},
    value=64.76,
    domain={'x': [0.1, 1], 'y': [0.2, 0.9]},
    title={'text': "Avg Charges"}))

fig.show()

# %%
# KPI-3
df3 = pd.DataFrame(data, columns=['TotalCharges'])

av_column = df3.mean(axis=0)
print(av_column)

fig = go.Figure(go.Indicator(
    mode="number+delta",
    value=2283.30,
    delta={"valueformat": ".0f"},
    title={"text": "Avg TotalCharges"},
    domain={'y': [0, 1], 'x': [0.25, 0.75]}))

fig.add_trace(go.Scatter(
    y=[325, 324, 405, 400, 424, 404, 417, 432, 419, 394, 410, 426, 413, 419, 404, 408, 401, 377, 368, 361, 356, 359, 375, 397, 394, 418, 437, 450, 430, 442, 424, 443, 420, 418, 423, 423, 426, 440, 437, 436, 447, 460, 478, 472, 450, 456, 436, 418, 429, 412, 429, 442, 464, 447, 434, 457, 474, 480, 499, 497, 480, 502, 512, 492]))

fig.update_layout(xaxis={'range': [0, 62]})
fig.show()

# %%
# Fetching count of the target variables
data['Churn'].value_counts().plot(kind='bar')

# %%
# Distribution of TotalCharges
sns.distplot(data['TotalCharges'])

# %%
# Get all features
features = [column_name for column_name in data.columns if column_name != 'Churn']

# %%
# Get all categorical features
categorical = [column_name for column_name in features if data[column_name].dtype == 'object']

# %%
# Get all numeric columns
numeric = [column_name for column_name in features if column_name not in categorical]

print(features)
print('-------------------------------------')
print(categorical)
print('-------------------------------------')
print(numeric)

# %%
# Countplot for categorical variables
plt.rcParams["axes.labelsize"] = 5 
sns.set(font_scale=5)
fig, axes = plt.subplots(5, 3, figsize=(100, 100))

for ax, column in zip(axes.flatten(), categorical):
    # Create countplot
    sns.countplot(x=column, hue='Churn', data=data, ax=ax)
    
    # Set the title of each subplot
    ax.set_title(column)
    
    # Improve legends
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='right', fontsize=48)
    ax.get_legend().remove()

# %%
# Box plot for continuous variables
plt.rcParams["axes.labelsize"] = 1 
sns.set(font_scale=1)
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
for ax, column in zip(axes.flatten(), numeric):
    # Create a boxplot
    sns.boxplot(x="Churn", y=column, data=data, ax=ax)
    
    # Set title
    ax.set_title(column)

# %%
# Splitting data into continuous and categorical values assuming categorical values have <10 unique values
categorical_val = []
continuous_val = []
for column in data.columns:
    print('==============================')
    print(f"{column} : {data[column].unique()}")
    if len(data[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continuous_val.append(column)

print(categorical_val)
print('=============') 
print(continuous_val)

# %%
# Fetching the count of categorical columns
def cnt():
    for i in categorical_val:
        i = data[i]
        count = i.value_counts()
        print(count)
        print('----------')

cnt()

# %%
# Collapse 'dsl' into 'DSL'
data['InternetService'] = data['InternetService'].replace({'dsl': "DSL"})
data['InternetService'].unique()

# %%
# Changing the values to numerical as the model only accepts numerical inputs
data['gender'] = data['gender'].replace(['Male', 'Female'], [1, 2])
data['SeniorCitizen'] = data['SeniorCitizen'].replace(['Yes', 'No'], [1, 0])
data['Partner'] = data['Partner'].replace(['Yes', 'No'], [1, 0])
data['Dependents'] = data['Dependents'].replace(['Yes', 'No'], [1, 0])
data['PhoneService'] = data['PhoneService'].replace(['Yes', 'No'], [1, 0])
data['PaperlessBilling'] = data['PaperlessBilling'].replace(['Yes', 'No'], [1, 0])
data['Churn'] = data['Churn'].replace(['Stayed', 'Churned'], [1, 0])
data['MultipleLines'] = data['MultipleLines'].replace(['Yes', 'No', 'No phone service'], [1, 0, 2])
data['InternetService'] = data['InternetService'].replace(['Fiber optic', 'DSL', 'No'], [1, 2, 0])
data['OnlineSecurity'] = data['OnlineSecurity'].replace(['Yes', 'No', 'No internet service'], [1, 0, 2])
data['OnlineBackup'] = data['OnlineBackup'].replace(['Yes', 'No', 'No internet service'], [1, 0, 2])
data['DeviceProtection'] = data['DeviceProtection'].replace(['Yes', 'No', 'No internet service'], [1, 0, 2])
data['TechSupport'] = data['TechSupport'].replace(['Yes', 'No', 'No internet service'], [1, 0, 2])
data['StreamingTV'] = data['StreamingTV'].replace(['Yes', 'No', 'No internet service'], [1, 0, 2])
data['StreamingMovies'] = data['StreamingMovies'].replace(['Yes', 'No', 'No internet service'], [1, 0, 2])

data.head(5)

# %%
# Splitting Data
X = data.drop(columns=['Churn'])
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# %%
# Training the Model using Different Classifiers
def accuracy(estimator, name):
    model = estimator.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy score of {name} classifier: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion matrix of {name} classifier: \n{confusion_matrix(y_test, y_pred)}")

# KNN Classifier
accuracy(KNeighborsClassifier(), 'KNN')

# %%
# SVM Classifier
accuracy(SVC(), 'SVM')

# %%
# Decision Tree Classifier
accuracy(DecisionTreeClassifier(), 'Decision Tree')

# %%
# Random Forest Classifier
accuracy(RandomForestClassifier(), 'Random Forest')

# %%
# MLP Classifier
accuracy(MLPClassifier(), 'MLP')

# %%
# Naive Bayes Classifier
accuracy(GaussianNB(), 'Naive Bayes')

# %%
# Saving the Final Model (for example, Decision Tree)
final_model = DecisionTreeClassifier()
final_model.fit(X_train, y_train)
pickle.dump(final_model, open('decision_tree_model.pkl', 'wb'))
