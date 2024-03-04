'''
1- Creamos el entorno virtual: $ conda create --name mlflow python
2- Activamos: conda activate mlflow.
3- El python instalado es la version 3.12.3
4- pip install mlflow # instala varias librerias, instalo mlflow 2.11.0
'''
import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Iris dataset
X,y = datasets.load_iris(return_X_y = True)
print(X.shape)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyper parameters.
params ={
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}


# Train the model
lr =  LogisticRegression(**params)
lr.fit(X_train,y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test,y_pred)
print("accuracy",accuracy)



#mlflow.set_tracking_uri(uri="http://localhost:5000")

