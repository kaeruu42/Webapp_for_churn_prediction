import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('churn.csv')

data['TotalCharges'] = data['TotalCharges'].apply(pd.to_numeric, errors='coerce')

# make boolean columns into binary
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
data['Partner'] = data['Partner'].map({'Yes': 1, 'No': 0})
data['Dependents'] = data['Dependents'].map({'Yes': 1, 'No': 0})
data['PhoneService'] = data['PhoneService'].map({'Yes': 1, 'No': 0})
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data['MultipleLines'] = data['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 0})
data['OnlineSecurity'] = data['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
data['OnlineBackup'] = data['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
data['DeviceProtection'] = data['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
data['TechSupport'] = data['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
data['StreamingTV'] = data['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
data['StreamingMovies'] = data['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes': 1, 'No': 0})
data = data.drop(['customerID'], axis=1)

# splitting data into x(independent) and y(dependent) variables
X = data.iloc[:, 0:19]
y = data.iloc[:, 19]

#features = X.columns
# One Hot Encoding (changing categories/strings into numbers)
X = pd.get_dummies(X)

features = X.columns

# Imputation for the null values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X)
X = imputer.transform(X)

# splitting the data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature scaling
# converts X variables into values ranging from -1 to +1
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)
# x_train = pd.DataFrame(x_train)

# Modeling
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Training accuracy: ", model.score(x_train, y_train))
print("Testing accuracy: ", model.score(x_test, y_test))

cm = confusion_matrix(y_test, y_pred)
print(cm)

# save model
# pickle.dump(model, open('model.pkl', 'wb'))


# np.set_printoptions(threshold=np.inf)
# print(y_pred)
