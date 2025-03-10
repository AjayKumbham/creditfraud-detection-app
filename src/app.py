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

# Load data
data = pd.read_csv('data/Fraud Detection Dataset.csv')

# Create risk score function
def calculate_risk_score(row):
    score = 0
    
    # Amount risk
    if row['Transaction_Amount'] > 10000: score += 3
    elif row['Transaction_Amount'] > 5000: score += 2
    elif row['Transaction_Amount'] > 1000: score += 1
    
    # Time risk (0-23)
    if 0 <= row['Time_of_Transaction'] <= 5: score += 2  # Midnight to 5AM
    
    # Previous fraud risk
    if row['Previous_Fraudulent_Transactions'] > 0: score += 3
    
    # Account age risk
    if row['Account_Age'] < 30: score += 2  # New accounts
    
    # Transaction frequency risk
    if row['Number_of_Transactions_Last_24H'] > 20: score += 2
    
    # Location risk
    if row['Location'] == 'Foreign': score += 2
    
    # Device risk for ATM
    if row['Transaction_Type'] == 'ATM Withdrawal' and row['Device_Used'] == 'Mobile':
        score += 2
    
    return score > 5  # Convert to binary: 1 if high risk, 0 if low risk

# Prepare features with risk scoring
X = pd.DataFrame({
    'Amount': data['Transaction_Amount'].fillna(data['Transaction_Amount'].mean()),
    'Time': data['Time_of_Transaction'].fillna(data['Time_of_Transaction'].mean()),
    'PrevFraud': data['Previous_Fraudulent_Transactions'] * 3,  # Higher weight
    'AccountAge': data['Account_Age'],
    'Transactions24H': data['Number_of_Transactions_Last_24H'],
    'IsATM': (data['Transaction_Type'] == 'ATM Withdrawal').astype(int) * 2,
    'IsForeign': (data['Location'] == 'Foreign').astype(int) * 2,
    'IsMobile': (data['Device_Used'] == 'Mobile').astype(int),
    'RiskScore': data.apply(calculate_risk_score, axis=1) * 2  # Add risk score as feature
})

# Target variable
y = data['Fraudulent']

# Train model with stricter parameters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight={0: 1, 1: 3},  # Much higher weight on fraud
    random_state=42
)
model.fit(X_scaled, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")
        
        # Calculate risk score for input
        input_data = pd.Series({
            'Transaction_Amount': float(data['amount']),
            'Time_of_Transaction': float(data['time']),
            'Previous_Fraudulent_Transactions': int(data['prev_fraud']),
            'Account_Age': int(data['account_age']),
            'Number_of_Transactions_Last_24H': int(data['transactions_24h']),
            'Transaction_Type': data['type'],
            'Device_Used': data['device'],
            'Location': data['location']
        })
        risk_score = calculate_risk_score(input_data)
        
        # Create feature vector with risk score
        features = np.array([[
            float(data['amount']),
            float(data['time']),
            int(data['prev_fraud']) * 3,
            int(data['account_age']),
            int(data['transactions_24h']),
            2 if data['type'] == 'ATM Withdrawal' else 0,
            2 if data['location'] == 'Foreign' else 0,
            1 if data['device'] == 'Mobile' else 0,
            risk_score * 2
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction with risk score adjustment
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Force high probability for very risky transactions
        if risk_score and probability < 0.7:
            probability = max(probability, 0.7)
        
        logger.info(f"Risk Score: {risk_score}, Prediction: {prediction}, Probability: {probability:.2%}")
        
        # Calculate risk factors
        feature_names = [
            'Transaction Amount', 
            'Time of Day', 
            'Previous Frauds', 
            'Account Age', 
            'Transactions in 24H',
            'ATM Withdrawal',
            'Foreign Location', 
            'Mobile Device',
            'Risk Score'
        ]
        importances = dict(zip(feature_names, model.feature_importances_))
        
        response = {
            'status': 'success',
            'prediction': 'Fraudulent' if prediction == 1 or risk_score else 'Legitimate',
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
                    'transactions_24h': int(data['transactions_24h']),
                    'risk_score': 'High' if risk_score else 'Low'
                },
                'feature_importance': importances
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
