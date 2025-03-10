# Credit Card Fraud Detection System 🛡️

A sophisticated machine learning system that detects fraudulent credit card transactions in real-time. Built with Python, Flask, and scikit-learn, this application provides instant risk analysis and detailed insights for each transaction.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0.2-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.1-orange.svg)](https://scikit-learn.org/)

## 📊 About the Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, created by the Machine Learning Group at ULB (Université Libre de Bruxelles). 

**Dataset Characteristics:**
- Contains real credit card transactions made in September 2013
- 284,807 transactions, with only 492 frauds (0.172% fraudulent)
- Highly unbalanced dataset reflecting real-world scenarios
- Features are numerical and PCA-transformed for confidentiality
- Time, Amount, and 28 principal components (V1-V28)

**Credit:** 
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. "Calibrating Probability with Undersampling for Unbalanced Classification." In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

## 🌟 Features

### Real-Time Analysis
- Instant transaction risk assessment
- Comprehensive fraud probability scoring
- Detailed breakdown of risk factors
- Visual risk indicators and alerts

### Risk Assessment Factors
- Transaction amount patterns
- Location-based risk analysis
- Authentication method evaluation
- Merchant history analysis
- Time-based pattern recognition
- Card usage behavior

### User Interface
- Clean, professional design
- Interactive form with detailed inputs
- Real-time feedback
- Risk visualization
- Hidden test scenarios for demonstration

## 🔧 Technical Architecture

### Frontend
- **HTML5/Bootstrap** for responsive design
- **JavaScript** for dynamic updates
- Interactive risk visualization
- Real-time form validation

### Backend
- **Flask** web framework
- **scikit-learn** for ML model
- **pandas** for data processing
- **numpy** for numerical operations

### Machine Learning
- **Algorithm**: Random Forest Classifier
- **Feature Engineering**: 
  - Time-based features
  - Amount normalization
  - Location risk scoring
  - Authentication risk assessment
- **Model Performance**:
  - High precision for fraud detection
  - Low false positive rate
  - Real-time prediction capability

## 🚀 Project Structure
```
project/
├── src/
│   └── app.py              # Main Flask application
├── models/                 # Trained models and feature data
│   ├── fraud_model.pkl     # Trained Random Forest model
│   ├── fraud_scaler.pkl    # Feature scaler
│   └── feature_info.pkl    # Feature descriptions
├── data/                   # Dataset directory
│   └── creditcard.csv      # Credit card transaction dataset
├── templates/              # HTML templates
│   └── index.html         # Main application interface
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/AjayKumbham/creditfraud-detection-app.git
cd creditfraud-detection-app
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
   - Visit [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv`
   - Place it in the `data/` directory

## 🎮 Usage

1. Start the Flask server:
```bash
python src/app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## 📝 Transaction Risk Levels

### Low Risk (Green)
- Local transactions
- Regular merchant
- Strong authentication
- Normal amount range
- Typical time and location

### Medium Risk (Yellow)
- Different state/location
- New merchant
- Online transaction
- Higher than usual amount
- Unusual timing

### High Risk (Red)
- International location
- First-time merchant
- Weak authentication
- Very high amount
- Suspicious timing

## 🧪 Test Scenarios

Access test scenarios through the "Test Options" button:

### 1. Low Risk Example
- Amount: $75
- Location: Local grocery store
- Authentication: Chip & PIN
- Time: 2:00 PM
- Expected: Low risk, instant approval

### 2. Medium Risk Example
- Amount: $500
- Location: Online shopping, different state
- Authentication: Online verification
- Time: 8:30 PM
- Expected: Medium risk, additional verification

### 3. High Risk Example
- Amount: $2,000
- Location: International
- Authentication: Manual entry
- Time: 3:00 AM
- Expected: High risk, likely decline

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset provided by the Machine Learning Group at ULB
- Kaggle for hosting the dataset
- scikit-learn team for the machine learning tools
- Flask team for the web framework

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository.
