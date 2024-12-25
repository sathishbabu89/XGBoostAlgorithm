import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
#data = pd.read_excel("G:\MachineLearning\XGBoostApp\fraud_detection_data.xlsx")  # Ensure the file is in the same folder as this script
data = pd.read_excel(r"G:\MachineLearning\XGBoostApp\fraud_detection_data.xlsx")

# Display the first few rows of the dataset
print("Dataset preview:")
print(data.head())

# Separate features and target variable
X = data.drop(["TransactionID", "IsFraud"], axis=1)  # Features
y = data["IsFraud"]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Predict fraud for a new transaction (example)
new_transaction = pd.DataFrame({
    "Amount": [250.75],
    "Location": [3],
    "TransactionType": [1],
    "TimeOfDay": [14]
})

predicted_class = model.predict(new_transaction)[0]
print(f"\nPredicted class for new transaction: {predicted_class} ({'Fraud' if predicted_class == 1 else 'Not Fraud'})")
