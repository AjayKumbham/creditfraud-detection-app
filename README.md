# Credit Card Fraud Detection System

A machine learning-based web application that detects potentially fraudulent credit card transactions using both ML predictions and a risk scoring system.

## Features

- Real-time transaction analysis
- Hybrid detection approach:
  - Machine Learning (Random Forest Classifier)
  - Risk Scoring System
- Risk factors analyzed:
  - Transaction amount
  - Time of transaction
  - Previous fraud history
  - Account age
  - Transaction frequency
  - Location (Local/Foreign)
  - Device used
  - Transaction type
  - Payment method

## Tech Stack

- Python 3.x
- Flask (Web Framework)
- scikit-learn (Machine Learning)
- pandas (Data Processing)
- HTML/CSS (Frontend)
- Bootstrap 5 (UI Framework)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AjayKumbham/creditfraud-detection-app.git
cd creditfraud-detection-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python src/app.py
```

4. Open http://localhost:5000 in your browser

## How It Works

The system uses a hybrid approach for fraud detection:

1. **Machine Learning Model**
   - Random Forest Classifier
   - Trained on historical transaction data
   - Features include transaction amount, time, location, etc.
   - Class weights adjusted for imbalanced data

2. **Risk Scoring System**
   - Rule-based scoring mechanism
   - Assigns risk points based on suspicious patterns
   - High-risk indicators:
     - Large transactions (>$10,000)
     - Late night transactions (12 AM - 5 AM)
     - Previous fraud history
     - New accounts (<30 days)
     - High transaction frequency
     - Foreign locations
     - Unusual device usage

3. **Combined Analysis**
   - ML model prediction
   - Risk score calculation
   - Final fraud probability
   - Detailed risk factor breakdown

## Usage

1. Enter transaction details:
   - Transaction type
   - Amount
   - Time
   - Device used
   - Location
   - Account details

2. Click "Analyze Transaction"

3. View results:
   - Fraud prediction
   - Risk probability
   - Contributing risk factors
   - Transaction analysis

## Project Structure

```
creditfraud-detection-app/
├── src/
│   └── app.py         # Main application code
├── templates/
│   └── index.html     # Web interface
├── data/
│   └── Fraud Detection Dataset.csv  # Training data
├── requirements.txt   # Dependencies
└── README.md         # Documentation
```

## Contributing

Feel free to submit issues and enhancement requests!
