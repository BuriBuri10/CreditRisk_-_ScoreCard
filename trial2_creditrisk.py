import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Hybrid Credit Risk Model", layout="wide")
st.title("🤖 Hybrid Credit Risk Model: XGBoost + LightGBM + Logistic Regression")

# Upload dataset
uploaded_file = st.file_uploader("Upload your credit risk dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Column mapping
    column_mapping = {
        'age': 'person_age',
        'income': 'person_income',
        'debt_to_income': None,  # Can be derived
        'loan_amount': 'loan_amnt',
        'loan_purpose': 'loan_intent',
        'missed_payments': 'cb_person_cred_hist_length',
        'total_debt': None,  # Not directly available
        'num_credit_inquiries': 'cb_person_default_on_file',  # Proxy for behavior
        'credit_score': 'credit_score'
    }

    required_features = [v for v in column_mapping.values() if v is not None]

    missing_features = [feature for feature in required_features if feature not in df.columns]
    if missing_features:
        st.warning(f"Missing features in dataset: {', '.join(missing_features)}")
        if 'person_income' in df.columns and 'cb_person_cred_hist_length' in df.columns:
            st.info("Suggested Feature: Debt-to-Income Ratio can be derived if total debt becomes available.")
    else:
        st.success("✅ All required features are present.")

    # Detect target column
    target_columns = ['default', 'is_default', 'loan_status']
    target_column = next((col for col in target_columns if col in df.columns), None)
    if target_column:
        st.success(f"Target column detected: {target_column}")
    else:
        st.error("No target column found. Make sure the dataset includes a binary target column.")
        st.stop()

    # Prepare feature matrix X and target vector y
    X = df[[col for col in required_features if col in df.columns]].copy()
    y = df[target_column]

    # Automatically infer types to separate numerical and categorical columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # # Identify numeric and categorical columns
    # categorical_columns = ['loan_intent', 'cb_person_default_on_file'] if 'loan_intent' in X.columns else []
    # numeric_columns = [col for col in X.columns if col not in categorical_columns]

    # Impute missing values
    numeric_imputer = SimpleImputer(strategy='mean')
    X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])

    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_columns:
        X[col] = le.fit_transform(X[col].astype(str))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # XGBoost
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_preds)

    # LightGBM
    lgb_model = lgb.LGBMClassifier()
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_preds)

    # Logistic Regression
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)
    log_reg_preds = log_reg_model.predict_proba(X_test)[:, 1]
    log_reg_auc = roc_auc_score(y_test, log_reg_preds)

    # Output AUCs
    st.subheader("📈 Model Performance")
    st.metric("XGBoost AUC", f"{xgb_auc:.4f}")
    st.metric("LightGBM AUC", f"{lgb_auc:.4f}")
    st.metric("Logistic Regression AUC", f"{log_reg_auc:.4f}")

    best_model = max([(xgb_auc, "XGBoost"), (lgb_auc, "LightGBM"), (log_reg_auc, "Logistic Regression")])
    st.info(f"Best performing model: {best_model[1]} with AUC: {best_model[0]:.4f}")

    # Feature Importance
    st.subheader("🔍 XGBoost Feature Importance")
    xgb_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.dataframe(xgb_importance)

    st.subheader("🔍 LightGBM Feature Importance")
    lgb_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': lgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.dataframe(lgb_importance)

    # Logistic Regression Coefficients
    st.subheader("📋 Logistic Regression Coefficients")
    log_reg_coef = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': log_reg_model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)
    st.dataframe(log_reg_coef)

    st.markdown("---")
    st.caption("This hybrid approach supports both performance and compliance needs.")
