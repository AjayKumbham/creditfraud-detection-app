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

def calculate_risk_score(transaction):
    """Calculate risk score based on rules"""
    score = 0
    reasons = []
    
    # Amount risk (max 3 points)
    if transaction['Transaction_Amount'] > 10000:
        score += 3
        reasons.append("Very high amount")
    elif transaction['Transaction_Amount'] > 5000:
        score += 2
        reasons.append("High amount")
    elif transaction['Transaction_Amount'] > 1000:
        score += 1
        reasons.append("Moderate amount")
    
    # Time risk (max 2 points)
    if 0 <= transaction['Time_of_Transaction'] <= 5:
        score += 2
        reasons.append("Late night transaction (12AM-5AM)")
    
    # Previous fraud risk (max 3 points)
    if transaction['Previous_Fraudulent_Transactions'] > 0:
        score += 3
        reasons.append("Previous fraud history")
    
    # Account age risk (max 2 points)
    if transaction['Account_Age'] < 30:
        score += 2
        reasons.append("New account (<30 days)")
    
    # Transaction frequency risk (max 2 points)
    if transaction['Number_of_Transactions_Last_24H'] > 20:
        score += 2
        reasons.append("High transaction frequency")
    
    # Location risk (max 2 points)
    if transaction['Location'] == 'Foreign':
        score += 2
        reasons.append("Foreign location")
    
    # Device risk for ATM (max 2 points)
    if transaction['Transaction_Type'] == 'ATM Withdrawal' and transaction['Device_Used'] == 'Mobile':
        score += 2
        reasons.append("Mobile ATM access")
    
    risk_level = "High" if score >= 8 else "Medium" if score >= 4 else "Low"
    
    return {
        'score': score,
        'max_score': 16,
        'risk_level': risk_level,
        'reasons': reasons
    }

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
        
        # Get ML prediction
        features = preprocess_data(input_df)
        features_scaled = scaler.transform(features)
        ml_prediction = model.predict(features_scaled)[0]
        ml_probability = model.predict_proba(features_scaled)[0][1]
        
        # Get risk score
        risk_analysis = calculate_risk_score(input_df.iloc[0])
        is_high_risk = risk_analysis['risk_level'] == "High"
        
        # Combine both systems
        final_prediction = "Fraudulent" if (ml_prediction == 1 or is_high_risk) else "Legitimate"
        
        # Adjust probability based on risk score
        final_probability = max(ml_probability, risk_analysis['score'] / risk_analysis['max_score'])
        
        # Get feature importance for ML model
        feature_importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
        
        # Log prediction details
        logger.info(f"ML Prediction: {ml_prediction}, Probability: {ml_probability:.2%}")
        logger.info(f"Risk Level: {risk_analysis['risk_level']}, Score: {risk_analysis['score']}")
        logger.info(f"Final Prediction: {final_prediction}, Final Probability: {final_probability:.2%}")
        
        response = {
            'status': 'success',
            'prediction': final_prediction,
            'fraud_probability': f'{final_probability:.1%}',
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
                'ml_analysis': {
                    'prediction': 'Fraudulent' if ml_prediction == 1 else 'Legitimate',
                    'probability': f'{ml_probability:.1%}',
                    'feature_importance': feature_importance
                },
                'risk_analysis': {
                    'level': risk_analysis['risk_level'],
                    'score': f'{risk_analysis["score"]}/{risk_analysis["max_score"]}',
                    'factors': risk_analysis['reasons']
                }
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
