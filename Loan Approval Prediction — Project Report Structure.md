# LOAN APPROVAL PREDICTION SYSTEM

## Project Report

**Submitted by:** YOUR NAME  
**Course:** YOUR COURSE  
**Institution:** YOUR COLLEGE / UNIVERSITY  
**Academic Year:** 2026  

---

# Table of Contents

1. Abstract
2. Introduction
3. Problem Statement
4. Project Objectives
5. Scope of the Project
6. Dataset Description
7. Technologies Used
8. System Architecture
9. Data Preprocessing
10. Exploratory Data Analysis
11. Feature Engineering
12. Machine Learning Methodology
13. Model Evaluation
14. Application / Dashboard
15. Business Insights
16. Limitations
17. Future Scope
18. Conclusion
19. References
20. Appendix

---

# 1. Abstract

Loan approval is an important decision-making process in the financial sector. This project develops a machine-learning-based Loan Approval Prediction System that analyzes historical loan application data and predicts whether a new loan application is likely to be approved or rejected.

The project includes data preprocessing, exploratory data analysis, feature engineering, machine learning model development, model evaluation, and an interactive Streamlit application.

Two classification algorithms, Logistic Regression and Random Forest, are evaluated using metrics including accuracy, precision, recall, and F1 score.

The final application allows users to enter applicant information and obtain a model-generated prediction together with an approval probability.

---

# 2. Introduction

Financial institutions receive loan applications containing information about applicants, their financial circumstances, and requested loans.

Analyzing historical applications can reveal patterns associated with loan approval and rejection.

This project applies data analytics and machine learning techniques to historical loan application data in order to develop an interactive preliminary loan prediction system.

The project demonstrates how raw data can be transformed into analysis, insights, and a usable application.

---

# 3. Problem Statement

Manual evaluation of loan applications can require the analysis of multiple applicant characteristics.

The problem addressed by this project is:

> **How can historical loan application data be analyzed and used to develop a machine-learning system that predicts the likely approval status of a new loan application?**

The project focuses on creating a data-driven prototype rather than replacing real-world banking or lending decisions.

---

# 4. Project Objectives

The objectives of this project are:

1. Collect and understand loan application data.
2. Clean and preprocess the dataset.
3. Analyze important characteristics of loan applications.
4. Identify patterns associated with loan approval.
5. Engineer useful features.
6. Train classification models.
7. Evaluate model performance.
8. Develop an interactive prediction interface.
9. Present useful business observations from the data.
10. Document the complete project in a reproducible format.

---

# 5. Scope of the Project

The project covers:

- Historical loan application analysis
- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Classification modelling
- Model evaluation
- Interactive prediction

The project does not attempt to implement a complete production banking system.

---

# 6. Dataset Description

## 6.1 Dataset Source

**Source:** INSERT FINAL DATASET URL

The dataset contains historical loan application information.

## 6.2 Dataset Features

| Feature | Description |
|---|---|
| Loan_ID | Unique loan application identifier |
| Gender | Applicant gender |
| Married | Marital status |
| Dependents | Number of dependents |
| Education | Applicant education status |
| Self_Employed | Self-employment status |
| ApplicantIncome | Applicant income |
| CoapplicantIncome | Coapplicant income |
| LoanAmount | Requested loan amount |
| Loan_Amount_Term | Loan repayment term |
| Credit_History | Credit history indicator |
| Property_Area | Property location category |
| Loan_Status | Loan approval status |

## 6.3 Target Variable

The prediction target is:

**Loan_Status**

```text
Y → Approved
N → Rejected
```

---

# 7. Technologies Used

The project uses:

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

GitHub is used for source-code repository management and project submission.

---

# 8. System Architecture

The overall architecture is:

```text
                    ┌─────────────────┐
                    │  Loan Dataset   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Data Cleaning   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      EDA        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │Feature Engineer │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Preprocessing   │
                    └────────┬────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │ Machine Learning Models │
                 └───────────┬─────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Model Evaluation│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Streamlit App   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Loan Prediction │
                    └─────────────────┘
```

**[INSERT SYSTEM ARCHITECTURE SCREENSHOT/DIAGRAM HERE]**

---

# 9. Data Preprocessing

## 9.1 Duplicate Removal

Duplicate records are identified and removed.

## 9.2 Missing Value Treatment

Numerical features are processed using median imputation.

Categorical features are processed using the most frequently occurring category.

## 9.3 Categorical Encoding

Categorical variables are converted into numerical representations using One-Hot Encoding.

## 9.4 Numerical Scaling

Numerical variables are standardized before being supplied to the machine-learning models.

---

# 10. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure and patterns in the dataset.

## 10.1 Loan Approval Distribution

**[INSERT APPROVAL/REJECTION CHART HERE]**

### Observation

Describe the observed distribution of approved and rejected applications.

---

## 10.2 Applicant Income Distribution

**[INSERT INCOME HISTOGRAM HERE]**

### Observation

Describe the distribution of applicant income and any noticeable concentration or outliers.

---

## 10.3 Credit History and Loan Approval

**[INSERT CREDIT HISTORY CHART HERE]**

### Observation

Describe the relationship observed between credit history and loan approval in the dataset.

---

## 10.4 Property Area Analysis

**[INSERT PROPERTY AREA CHART HERE]**

### Observation

Compare the observed approval rates across property-area categories.

---

## 10.5 Correlation Analysis

**[INSERT CORRELATION HEATMAP HERE]**

### Observation

Discuss important relationships among the numerical variables.

---

# 11. Feature Engineering

Two additional features are created.

## 11.1 Total Income

```text
TotalIncome =
ApplicantIncome + CoapplicantIncome
```

This combines applicant and coapplicant income.

## 11.2 Loan-Income Ratio

```text
LoanIncomeRatio =
LoanAmount / TotalIncome
```

This provides an additional measure of the requested loan relative to combined income.

---

# 12. Machine Learning Methodology

The project treats loan approval as a binary classification problem.

The target values are converted into:

```text
Approved → 1
Rejected → 0
```

The dataset is divided into training and testing subsets.

```text
Training Data → 80%
Testing Data  → 20%
```

A fixed random state is used to make the experiment reproducible.

---

# 13. Machine Learning Models

## 13.1 Logistic Regression

Logistic Regression is used as a baseline classification algorithm.

It estimates the probability of the target class based on the input features.

---

## 13.2 Random Forest

Random Forest is an ensemble machine-learning method based on multiple decision trees.

It is included to compare the performance of an ensemble model with Logistic Regression.

---

# 14. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 14.1 Model Comparison

Insert the actual results generated by the application:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | XX% | XX% | XX% | XX% |
| Random Forest | XX% | XX% | XX% | XX% |

**Important:** Replace `XX` with the actual values produced when you run the project. Do not invent the results before running the code.

---

## 14.2 Confusion Matrix

**[INSERT CONFUSION MATRIX SCREENSHOT HERE]**

Explain:

- True positives
- True negatives
- False positives
- False negatives

---

## 14.3 Classification Report

**[INSERT CLASSIFICATION REPORT SCREENSHOT HERE]**

Discuss the observed precision, recall, and F1 scores.

---

# 15. Application / Dashboard

The project includes an interactive Streamlit application.

## 15.1 Dashboard Page

The dashboard displays:

- Total applications
- Approved applications
- Rejected applications
- Approval rate
- Approval distribution
- Property-area analysis
- Key observations

**[INSERT DASHBOARD SCREENSHOT HERE]**

---

## 15.2 Data Analysis Page

The Data Analysis page provides:

- Dataset preview
- Missing-value analysis
- Income distribution
- Credit-history analysis
- Property-area analysis
- Correlation analysis

**[INSERT DATA ANALYSIS SCREENSHOT HERE]**

---

## 15.3 Model Performance Page

The Model Performance page provides:

- Model comparison
- Evaluation metrics
- Confusion matrix
- Classification report

**[INSERT MODEL PERFORMANCE SCREENSHOT HERE]**

---

## 15.4 Loan Prediction Page

The user can enter applicant details such as:

- Gender
- Married status
- Dependents
- Education
- Employment status
- Applicant income
- Coapplicant income
- Loan amount
- Loan term
- Credit history
- Property area

The application then produces a predicted loan status.

**[INSERT LOAN PREDICTION SCREENSHOT HERE]**

---

# 16. Business Insights

The analysis can be used to investigate several business questions.

### Insight 1 — Credit History

Describe the observed relationship between credit history and loan approval.

### Insight 2 — Income

Describe the observed relationship between applicant income, total income, and loan approval.

### Insight 3 — Loan Amount

Discuss how requested loan amount relates to applicant income.

### Insight 4 — Property Area

Describe differences in observed approval rates between property-area categories.

### Insight 5 — Model Predictions

Discuss where the model correctly identifies applications and where incorrect predictions occur.

> These observations should be written from the actual charts and model results generated by the project.

---

# 17. Risks and Limitations

The model has several limitations.

### Historical Data

The model learns from historical applications and may not reflect current lending policies.

### Data Quality

Missing values, outliers, and limited sample size can affect model performance.

### Model Limitations

A machine-learning prediction is not equivalent to a real-world financial decision.

### Bias and Fairness

Historical data may contain patterns that do not necessarily represent fair or appropriate lending decisions. Further fairness analysis would be required before real-world deployment.

---

# 18. Future Scope

The project can be extended through:

1. Hyperparameter optimization.
2. Cross-validation.
3. Additional classification algorithms.
4. Explainable AI.
5. Feature importance analysis.
6. Fairness and bias evaluation.
7. Model monitoring.
8. Database integration.
9. Cloud deployment.
10. Automated model retraining.

---

# 19. Conclusion

The Loan Approval Prediction System demonstrates how data analytics and machine learning can be applied to a loan-application dataset.

The project follows a complete workflow from data preparation and exploratory analysis to machine-learning modelling and interactive prediction.

The Streamlit application provides a practical interface for exploring the data and testing applicant information against the trained classification model.

The project demonstrates the transformation of raw data into analysis, machine-learning predictions, and a usable data-driven application.

---

# 20. References

Add the actual sources used for the project here.

### Dataset

**Dataset:** INSERT FINAL DATASET NAME  
**Source:** INSERT DATASET URL

### Documentation

- Python Documentation
- Pandas Documentation
- NumPy Documentation
- Scikit-learn Documentation
- Streamlit Documentation
- Matplotlib Documentation
- Seaborn Documentation

Add the URLs actually used during development.

---

# 21. Appendix

## Appendix A — Project Files

```text
loan_approval_prediction.py
requirements.txt
README.md
Loan_Approval_Project_Report.pdf
```

## Appendix B — Important Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run loan_approval_prediction.py
```

## Appendix C — GitHub Repository

**Repository URL:**

INSERT YOUR GITHUB REPOSITORY URL

## Appendix D — Screenshots

Include final screenshots of:

1. Dashboard
2. Data Analysis
3. Model Performance
4. Confusion Matrix
5. Loan Prediction
6. Prediction Result