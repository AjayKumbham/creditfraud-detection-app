# Credit Card Fraud Detection System

A machine learning-based system for real-time credit card fraud detection with an interactive web interface.

## Project Structure
```
project/
├── src/
│   └── app.py              # Main Flask application
├── models/                 # Trained models and feature data
├── data/                  # Dataset directory
├── templates/             # HTML templates
│   └── index.html
├── requirements.txt       # Project dependencies
└── README.md
```

## Setup Instructions

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Download the Credit Card Fraud Detection dataset from Kaggle
   - Place the `creditcard.csv` file in the `data/` directory

## Running the Application

1. Start the Flask server:
```bash
python src/app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Features

- Real-time transaction analysis
- Risk scoring based on multiple factors
- Detailed transaction insights
- Interactive web interface
- Support for various transaction types
- Quick test scenarios for demonstration

## Test Scenarios

The application includes pre-configured test scenarios accessible through the "Test Options" button:

1. Low Risk Scenario:
   - $75 grocery store purchase
   - Local transaction with chip & PIN

2. Medium Risk Scenario:
   - $500 online shopping
   - Different state transaction

3. High Risk Scenario:
   - $2,000 international transaction
   - Manual card entry

## Technology Stack

- Python 3.8+
- Flask 3.0.2
- scikit-learn 1.4.1
- pandas 2.2.1
- numpy 1.26.4

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided by the Machine Learning Group - ULB
- Scikit-learn team for ML tools
- Flask team for web framework
