# ============================================================
# LOAN APPROVAL PREDICTION PROJECT
# ============================================================
# This project:
# 1. Loads a loan approval dataset
# 2. Cleans the data
# 3. Performs exploratory data analysis
# 4. Trains Logistic Regression and Random Forest models
# 5. Evaluates the models
# 6. Provides a Streamlit prediction interface
#
# Run:
#     streamlit run loan_approval_prediction.py
#
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Loan Approval Prediction System")

st.markdown("""
### Machine Learning Based Loan Eligibility Analysis

This application analyzes historical loan applications and predicts
whether a new loan application is likely to be approved or rejected.
""")


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    # --------------------------------------------------------
    # OPTION 1:
    # If the user already has a CSV, place it in the project
    # folder with this name.
    # --------------------------------------------------------

    local_file = Path(__file__).resolve().parent / "loan_prediction.csv"

    if os.path.exists(local_file):
        df = pd.read_csv(local_file)
        return df

    # --------------------------------------------------------
    # OPTION 2:
    # Download a public copy automatically.
    #
    # This is a public GitHub copy of the commonly used
    # Loan Prediction dataset.
    # --------------------------------------------------------

    url = (
        "https://raw.githubusercontent.com/"
        "Nwaneto/44c4743cd97b8fb44fed5330cfb637a4/"
        "master/loan-prediction-dataset.csv"
    )

    try:
        df = pd.read_csv(url)
        return df

    except Exception:
        st.error(
            "Dataset could not be downloaded. "
            "Please download the CSV manually and save it as "
            "'loan_prediction.csv' in the project folder."
        )

        return pd.DataFrame()


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()


if df.empty:
    st.stop()

if "Loan_Status" not in df.columns:
    st.error(
        "The dataset must contain a 'Loan_Status' column. "
        "Please use the expected Loan Prediction dataset."
    )
    st.stop()


# ============================================================
# BASIC DATA CLEANING
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates()


# Normalize column names
df.columns = df.columns.str.strip()


# Convert target
if "Loan_Status" in df.columns:

    df["Loan_Status"] = (
        df["Loan_Status"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({
            "Y": 1,
            "N": 0,
            "YES": 1,
            "NO": 0,
            "1": 1,
            "0": 0
        })
    )


# Convert Dependents
if "Dependents" in df.columns:

    df["Dependents"] = (
        df["Dependents"]
        .astype(str)
        .replace("3+", "3")
    )

    df["Dependents"] = pd.to_numeric(
        df["Dependents"],
        errors="coerce"
    )


# Convert numerical columns
numeric_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Dependents"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

if (
    "ApplicantIncome" in df.columns
    and "CoapplicantIncome" in df.columns
):

    df["TotalIncome"] = (
        df["ApplicantIncome"]
        + df["CoapplicantIncome"]
    )


if (
    "LoanAmount" in df.columns
    and "TotalIncome" in df.columns
):

    df["LoanIncomeRatio"] = (
        df["LoanAmount"]
        / df["TotalIncome"].replace(0, np.nan)
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Data Analysis",
        "Model Performance",
        "Loan Prediction"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Executive Dashboard")

    total_applications = len(df)

    approved = (
        df["Loan_Status"].sum()
        if "Loan_Status" in df.columns
        else 0
    )

    rejected = total_applications - approved

    approval_rate = (
        approved / total_applications * 100
        if total_applications > 0
        else 0
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Applications",
        total_applications
    )

    col2.metric(
        "Approved",
        int(approved)
    )

    col3.metric(
        "Rejected",
        int(rejected)
    )

    col4.metric(
        "Approval Rate",
        f"{approval_rate:.2f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # APPROVAL DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Loan Approval Distribution")

        status_counts = df["Loan_Status"].value_counts()

        fig, ax = plt.subplots()

        ax.bar(
            ["Rejected", "Approved"],
            [
                status_counts.get(0, 0),
                status_counts.get(1, 0)
            ]
        )

        ax.set_ylabel("Number of Applications")
        ax.set_title("Loan Approval Distribution")

        st.pyplot(fig)

    # --------------------------------------------------------
    # PROPERTY AREA
    # --------------------------------------------------------

    with col2:

        if "Property_Area" in df.columns:

            st.subheader("Applications by Property Area")

            area_counts = df["Property_Area"].value_counts()

            fig, ax = plt.subplots()

            ax.bar(
                area_counts.index,
                area_counts.values
            )

            ax.set_xlabel("Property Area")
            ax.set_ylabel("Applications")

            st.pyplot(fig)

    # --------------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------------

    st.subheader("💡 Key Business Insights")

    if "Credit_History" in df.columns:

        credit_approval = (
            df.groupby("Credit_History")["Loan_Status"]
            .mean()
            .mul(100)
        )

        if 1 in credit_approval.index:

            st.write(
                f"• Applicants with a positive credit history "
                f"have an observed approval rate of "
                f"**{credit_approval[1]:.2f}%**."
            )

        if 0 in credit_approval.index:

            st.write(
                f"• Applicants with a negative credit history "
                f"have an observed approval rate of "
                f"**{credit_approval[0]:.2f}%**."
            )

    if "Education" in df.columns:

        education_approval = (
            df.groupby("Education")["Loan_Status"]
            .mean()
            .mul(100)
        )

        st.write(
            "• Approval rates can also be compared "
            "across education groups."
        )

        st.dataframe(
            education_approval
            .round(2)
            .rename("Approval Rate (%)")
        )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "Data Analysis":

    st.header("🔎 Exploratory Data Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    st.subheader("Missing Values")

    missing = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    if len(missing) > 0:
        st.dataframe(
            missing.rename("Missing Values")
        )
    else:
        st.success("No missing values found.")

    # --------------------------------------------------------
    # INCOME DISTRIBUTION
    # --------------------------------------------------------

    if "ApplicantIncome" in df.columns:

        st.subheader("Applicant Income Distribution")

        fig, ax = plt.subplots()

        ax.hist(
            df["ApplicantIncome"].dropna(),
            bins=30
        )

        ax.set_xlabel("Applicant Income")
        ax.set_ylabel("Number of Applicants")

        st.pyplot(fig)

    # --------------------------------------------------------
    # CREDIT HISTORY VS LOAN STATUS
    # --------------------------------------------------------

    if (
        "Credit_History" in df.columns
        and "Loan_Status" in df.columns
    ):

        st.subheader(
            "Credit History vs Loan Approval"
        )

        credit_table = pd.crosstab(
            df["Credit_History"],
            df["Loan_Status"]
        )

        credit_table.columns = [
            "Rejected",
            "Approved"
        ]

        st.dataframe(credit_table)

    # --------------------------------------------------------
    # PROPERTY AREA
    # --------------------------------------------------------

    if (
        "Property_Area" in df.columns
        and "Loan_Status" in df.columns
    ):

        st.subheader(
            "Approval Rate by Property Area"
        )

        area_approval = (
            df.groupby("Property_Area")["Loan_Status"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots()

        ax.bar(
            area_approval.index,
            area_approval.values
        )

        ax.set_ylabel("Approval Rate (%)")

        st.pyplot(fig)

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.subheader("Numerical Feature Correlation")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] > 1:

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)


# ============================================================
# MODEL PREPARATION
# ============================================================

@st.cache_resource
def train_models(data):

    model_df = data.copy()

    # --------------------------------------------------------
    # Remove rows without target
    # --------------------------------------------------------

    model_df = model_df.dropna(
        subset=["Loan_Status"]
    )

    # --------------------------------------------------------
    # Remove identifier
    # --------------------------------------------------------

    if "Loan_ID" in model_df.columns:

        model_df = model_df.drop(
            columns=["Loan_ID"]
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = model_df["Loan_Status"]

    X = model_df.drop(
        columns=["Loan_Status"]
    )

    # --------------------------------------------------------
    # Feature types
    # --------------------------------------------------------

    categorical_features = (
        X.select_dtypes(
            include=["object"]
        ).columns.tolist()
    )

    numerical_features = (
        X.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    # --------------------------------------------------------
    # Numeric preprocessing
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
    }

    trained_models = {}

    results = []

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        trained_models[name] = pipeline

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    results_df = pd.DataFrame(results)

    return (
        trained_models,
        results_df,
        X_test,
        y_test
    )


# ============================================================
# TRAIN MODELS
# ============================================================

(
    trained_models,
    model_results,
    X_test,
    y_test
) = train_models(df)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

if page == "Model Performance":

    st.header("🤖 Model Performance")

    st.subheader(
        "Model Comparison"
    )

    display_results = model_results.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_results[column] = (
            display_results[column] * 100
        ).round(2)

    st.dataframe(
        display_results,
        use_container_width=True
    )

    # --------------------------------------------------------
    # SELECT MODEL
    # --------------------------------------------------------

    selected_model_name = st.selectbox(
        "Select Model",
        list(trained_models.keys())
    )

    selected_model = trained_models[
        selected_model_name
    ]

    predictions = selected_model.predict(
        X_test
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Rejected",
            "Approved"
        ],
        yticklabels=[
            "Rejected",
            "Approved"
        ],
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    st.pyplot(fig)

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Rejected",
            "Approved"
        ],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose()
    )


# ============================================================
# LOAN PREDICTION
# ============================================================

elif page == "Loan Prediction":

    st.header("🔮 Loan Approval Prediction")

    st.markdown(
        "Enter applicant information below."
    )

    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    with st.form("loan_form"):

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            married = st.selectbox(
                "Married",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                [0, 1, 2, 3]
            )

            education = st.selectbox(
                "Education",
                [
                    "Graduate",
                    "Not Graduate"
                ]
            )

            self_employed = st.selectbox(
                "Self Employed",
                ["Yes", "No"]
            )

            property_area = st.selectbox(
                "Property Area",
                [
                    "Urban",
                    "Semiurban",
                    "Rural"
                ]
            )

        with col2:

            applicant_income = st.number_input(
                "Applicant Income",
                min_value=0,
                value=5000,
                step=500
            )

            coapplicant_income = st.number_input(
                "Coapplicant Income",
                min_value=0,
                value=0,
                step=500
            )

            loan_amount = st.number_input(
                "Loan Amount",
                min_value=0.0,
                value=150.0,
                step=10.0
            )

            loan_term = st.number_input(
                "Loan Term (months)",
                min_value=12,
                value=360,
                step=12
            )

            credit_history = st.selectbox(
                "Credit History",
                [1, 0],
                format_func=lambda x:
                    "Good" if x == 1
                    else "Poor"
            )

        submitted = st.form_submit_button(
            "Predict Loan Approval"
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if submitted:

        input_data = pd.DataFrame({
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [
                applicant_income
            ],
            "CoapplicantIncome": [
                coapplicant_income
            ],
            "LoanAmount": [
                loan_amount
            ],
            "Loan_Amount_Term": [
                loan_term
            ],
            "Credit_History": [
                credit_history
            ],
            "Property_Area": [
                property_area
            ]
        })

        # Feature engineering
        input_data["TotalIncome"] = (
            input_data["ApplicantIncome"]
            + input_data["CoapplicantIncome"]
        )

        input_data["LoanIncomeRatio"] = (
            input_data["LoanAmount"]
            / input_data["TotalIncome"].replace(
                0,
                np.nan
            )
        )

        # ----------------------------------------------------
        # Use Random Forest
        # ----------------------------------------------------

        model = trained_models[
            "Random Forest"
        ]

        prediction = model.predict(
            input_data
        )[0]

        probability = (
            model.predict_proba(
                input_data
            )[0][1]
        )

        st.divider()

        if prediction == 1:

            st.success(
                "### ✅ Loan is predicted as APPROVED"
            )

        else:

            st.error(
                "### ❌ Loan is predicted as REJECTED"
            )

        st.metric(
            "Approval Probability",
            f"{probability * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Applicant summary
        # ----------------------------------------------------

        st.subheader(
            "Applicant Summary"
        )

        summary = pd.DataFrame({
            "Feature": [
                "Applicant Income",
                "Coapplicant Income",
                "Loan Amount",
                "Loan Term",
                "Credit History",
                "Property Area",
                "Education"
            ],

            "Value": [
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_term,
                "Good"
                if credit_history == 1
                else "Poor",
                property_area,
                education
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Loan Approval Prediction | "
    "Data Analytics & Machine Learning Project"
)