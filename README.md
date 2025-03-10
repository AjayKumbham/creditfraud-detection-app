# Credit Card Fraud Detection System

A machine learning-based web application that detects potentially fraudulent credit card transactions using a Random Forest Classifier.

## Features

- Real-time transaction analysis
- Machine Learning based detection using Random Forest
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

The system uses a Random Forest Classifier for fraud detection:

1. **Data Preprocessing**
   - Consistent feature engineering
   - Standard scaling of numerical features
   - Binary encoding of categorical features
   - Missing value handling

2. **Model Features**
   - Transaction Amount
   - Time of Transaction
   - Previous Fraudulent Transactions
   - Account Age
   - Number of Transactions (24h)
   - ATM Transaction Flag
   - Foreign Location Flag
   - Mobile Device Flag

3. **Model Configuration**
   - Random Forest with 200 trees
   - Max depth of 8
   - Class weight balancing for fraud detection
   - Feature importance analysis

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
   - Fraud prediction
   - Probability score
   - Feature importance analysis
   - Transaction details

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
