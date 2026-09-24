# Loan Approval Prediction

A Streamlit application that explores historical loan applications, compares
two classification models, and predicts whether a new application is likely
to be approved.

## Features

- Dashboard with application totals, approval rate, and business insights
- Exploratory data analysis and missing-value checks
- Logistic Regression and Random Forest model comparison
- Confusion matrix and classification report
- Interactive loan prediction form

## Requirements

- Python 3.10 or newer
- Internet access on the first run, unless `loan_prediction.csv` is placed in
  the project folder

## Run the application

From this folder, create and activate a virtual environment, then install the
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run loan_approval_prediction.py
```

Streamlit will open the application in your browser. If it does not, open the
URL printed in the terminal, usually `http://localhost:8501`.

## Dataset

The app first looks for `loan_prediction.csv` beside the Python file. If it is
not present, it downloads a public copy of the common Loan Prediction dataset.
The dataset should include a `Loan_Status` target column and fields such as
`ApplicantIncome`, `LoanAmount`, `Credit_History`, and `Property_Area`.

## Project files

- `loan_approval_prediction.py` - Streamlit application and ML pipeline
- `requirements.txt` - Python dependencies
- `Loan Approval Prediction Project Report.pdf` - project report

## Notes

This project is for educational analysis only. A model prediction should not
be used as the sole basis for a real lending decision.