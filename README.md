# Credit Card Fraud Detection System

A hybrid fraud detection system that combines machine learning predictions with rule-based risk scoring for accurate fraud detection.

## Dataset

This project uses the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/ranjitmandal/fraud-detection-dataset-csv) from Kaggle, created by Ranjit Mandal. The dataset includes various transaction features that help in identifying fraudulent activities.

**Dataset Features:**
- Transaction amount
- Time of transaction
- Previous fraud history
- Account age
- Transaction frequency
- Location information
- Device information
- Transaction type

## Features

- Real-time transaction analysis
- Hybrid detection approach:
  - Machine Learning (Random Forest Classifier)
  - Rule-based Risk Scoring System
- Feature analysis:
  - Transaction amount
  - Time of transaction
  - Previous fraud history
  - Account age
  - Transaction frequency
  - Transaction type (ATM/Online)
  - Location (Local/Foreign)
  - Device used

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

The system uses a hybrid approach combining ML and rule-based detection:

1. **Machine Learning Model**
   - Random Forest Classifier
   - Standard scaling of numerical features
   - Binary encoding of categorical features
   - Feature importance analysis
   - Class weight balancing

2. **Risk Scoring System**
   - Rule-based scoring mechanism
   - Assigns risk points based on:
     - Large transactions (>$10,000)
     - Late night transactions (12 AM - 5 AM)
     - Previous fraud history
     - New accounts (<30 days)
     - High transaction frequency
     - Foreign locations
     - Mobile ATM access

3. **Combined Analysis**
   - ML model prediction
   - Risk score calculation
   - Final fraud probability
   - Detailed risk factor breakdown

## Usage

1. Enter transaction details:
   - Transaction amount
   - Time (0-23 hours)
   - Previous frauds
   - Account age (days)
   - Recent transactions
   - Transaction type
   - Location
   - Device used

2. Click "Analyze Transaction"

3. View results:
   - Final fraud prediction
   - ML model prediction
   - Risk score analysis
   - Contributing risk factors
   - Feature importance

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
## Screenshots

### 1. Legitimate Transaction Input  
User enters transaction details that are classified as legitimate.  
![Legitimate Transaction Input](screenshots/ss1.png)

### 2. Legitimate Transaction Analysis  
System confirms the transaction as legitimate and provides insights.  
![Legitimate Transaction Analysis](screenshots/ss2.png)

### 3. Fraudulent Transaction Input  
User enters transaction details that are flagged as fraudulent.  
![Fraudulent Transaction Input](screenshots/ss3.png)

### 4. Fraudulent Transaction Analysis  
System detects fraud, provides probability score, and highlights risk factors.  
![Fraudulent Transaction Analysis](screenshots/ss4.png)

## Credits

- Dataset provided by [Ranjit Mandal on Kaggle](https://www.kaggle.com/datasets/ranjitmandal/fraud-detection-dataset-csv)
- Built with scikit-learn and Flask

## Contributing

Feel free to submit issues and enhancement requests!
