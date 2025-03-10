import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../templates')

# Load and preprocess data
data = pd.read_csv('data/Fraud Detection Dataset.csv')

# Define feature columns for consistency
FEATURE_COLUMNS = [
    'Transaction_Amount',
    'Time_of_Transaction',
    'Previous_Fraudulent_Transactions',
    'Account_Age',
    'Number_of_Transactions_Last_24H',
    'Is_ATM',
    'Is_Foreign',
    'Is_Mobile'
]

def preprocess_data(df):
    """Preprocess data consistently for both training and prediction"""
    features = pd.DataFrame()
    
    # Numerical features - same order as FEATURE_COLUMNS
    features['Transaction_Amount'] = df['Transaction_Amount'].fillna(df['Transaction_Amount'].mean())
    features['Time_of_Transaction'] = df['Time_of_Transaction'].fillna(df['Time_of_Transaction'].mean())
    features['Previous_Fraudulent_Transactions'] = df['Previous_Fraudulent_Transactions']
    features['Account_Age'] = df['Account_Age']
    features['Number_of_Transactions_Last_24H'] = df['Number_of_Transactions_Last_24H']
    
    # Binary features - same order as FEATURE_COLUMNS
    features['Is_ATM'] = (df['Transaction_Type'] == 'ATM Withdrawal').astype(int)
    features['Is_Foreign'] = (df['Location'] == 'Foreign').astype(int)
    features['Is_Mobile'] = (df['Device_Used'] == 'Mobile').astype(int)
    
    return features[FEATURE_COLUMNS]

# Prepare training data
X = preprocess_data(data)
y = data['Fraudulent']

# Initialize and fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight={0: 1, 1: 3},  # Weight fraudulent cases more heavily
    random_state=42
)
model.fit(X_scaled, y)

logger.info("Model training completed")
logger.info(f"Feature importance: {dict(zip(FEATURE_COLUMNS, model.feature_importances_))}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")
        
        # Create DataFrame with single row for prediction
        input_df = pd.DataFrame({
            'Transaction_Amount': [float(data['amount'])],
            'Time_of_Transaction': [float(data['time'])],
            'Previous_Fraudulent_Transactions': [int(data['prev_fraud'])],
            'Account_Age': [int(data['account_age'])],
            'Number_of_Transactions_Last_24H': [int(data['transactions_24h'])],
            'Transaction_Type': [data['type']],
            'Location': [data['location']],
            'Device_Used': [data['device']]
        })
        
        # Preprocess input using same function as training
        features = preprocess_data(input_df)
        features_scaled = scaler.transform(features)
        
        # Get prediction and probability
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Get feature importance for this prediction
        feature_importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
        
        # Log prediction details
        logger.info(f"Features: {features.iloc[0].to_dict()}")
        logger.info(f"Prediction: {prediction}, Probability: {probability:.2%}")
        
        response = {
            'status': 'success',
            'prediction': 'Fraudulent' if prediction == 1 else 'Legitimate',
            'fraud_probability': f'{probability:.1%}',
            'analysis': {
                'transaction_details': {
                    'type': data['type'],
                    'amount': float(data['amount']),
                    'time': float(data['time']),
                    'device': data['device'],
                    'location': data['location'],
                    'payment_method': data['payment_method']
                },
                'account_analysis': {
                    'age': int(data['account_age']),
                    'prev_fraud': int(data['prev_fraud']),
                    'transactions_24h': int(data['transactions_24h'])
                },
                'feature_importance': feature_importance
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
