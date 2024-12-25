import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load the Iris dataset
data = load_iris()
X = data.data  # Features: Sepal length, Sepal width, Petal length, Petal width
y = data.target  # Target: Iris species (0, 1, 2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the datasets into XGBoost DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the parameters for XGBoost
params = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': 3,  # Number of classes
    'max_depth': 3,  # Maximum depth of a tree
    'eta': 0.3,  # Learning rate
    'seed': 42  # Random seed
}

# Train the model
num_rounds = 50  # Number of boosting rounds
model = xgb.train(params, dtrain, num_rounds)

# Make predictions
preds = model.predict(dtest)

# Evaluate the model
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy:.2f}")

# Predict on new data
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example input
dnew = xgb.DMatrix(new_data)
new_pred = model.predict(dnew)
print(f"Predicted class: {int(new_pred[0])} (Iris-{data.target_names[int(new_pred[0])]})")
