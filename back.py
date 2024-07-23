import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
from urllib.parse import urlparse
import tldextract  # Make sure to install this package

# Load your dataset
data = pd.read_csv('C:\\dsi\\enhanced_malicious_phish.csv')

# Check the columns of the DataFrame
print("Columns in the dataset:", data.columns)

# Use 'type' as the target column
target_column = 'type'
data = data.dropna(subset=[target_column])  # Drop rows with NaN in the target column
X = data.drop(columns=[target_column, 'url'])
y = data[target_column]

# Drop NaN in features
X = X.dropna()  
y = y[X.index]  

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average CV score:", cv_scores.mean())

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'random_forest_model.pkl')
print("Model saved as 'random_forest_model.pkl'")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# User input for URL classification
def preprocess_input_url(url):
    # Parse the URL
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(url)
    
    # Extract features
    features = {
        'url_length': [len(url)],  # Length of the URL
        'num_dots': [url.count('.')],  # Number of dots in the URL
        'num_special_chars': [sum(1 for char in url if char in ['-', '_', '?', '&'])],  # Number of special chars
        'token_count': [len(parsed_url.path.split('/')) if parsed_url.path else 0],  # Number of path tokens
        'domain_age': [1]  # Placeholder value, implement actual domain age calculation if needed
    }
    
    # Return DataFrame with the same columns as X_train
    return pd.DataFrame(features, columns=X_train.columns)

# Get user input
user_url = input("Enter the URL for analysis: ")

# Preprocess user URL to match training data format
X_user = preprocess_input_url(user_url)

# Ensure the feature names match
missing_cols = set(X_train.columns) - set(X_user.columns)
for col in missing_cols:
    X_user[col] = 0  # or some default value

# Make sure the columns are in the same order
X_user = X_user[X_train.columns]

# Make prediction
user_prediction = model.predict(X_user)
user_pred_label = label_encoder.inverse_transform(user_prediction)

# Check if the prediction indicates 'safe' or 'not safe'
if 'safe' in user_pred_label:
    result = "safe"
else:
    result = "not safe"

# Calculate overall model accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Prediction for the entered URL: {user_pred_label[0]} (This URL is considered {result})")
print(f"Overall Model Accuracy: {accuracy:.2f}")
print(f"Overall Model F1 Score: {f1:.2f}")
