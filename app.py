import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

model_list = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbor Classifier",
    "Naive Bayes Classifier",
    "Ensemble Model - Random Forest",
]


def calculate_display_metrics(y_test, y_pred, y_prob, model_name="Logistic Regression"):
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    matthews_corr = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(
        y_test, y_prob, multi_class="ovr"
    )

    st.subheader(f"{model_name} Model Evaluation Results")
    
    # 1. Structure your data into a dictionary
    metrics_data = {
        "Metric": ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC"],
        "Value": [accuracy, roc_auc, precision, recall, f1, matthews_corr]
    }

    # 2. Convert to a Pandas DataFrame
    df = pd.DataFrame(metrics_data)

    # 3. Render the interactive table (hiding the default index column)
    st.dataframe(df, hide_index=True, width=700, height=245)

    st.markdown("**Confusion Matrix:**")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=["Class 0", "Class 1", "Class 2"]
    )
    fig, ax = plt.subplots(figsize=(5, 3.5))
    disp.plot(cmap="Blues", ax=ax)
    plt.title(f"{model_name} Confusion Matrix")
    st.pyplot(fig)


def logistic_regression_prediction(X_test, y_test):
    logreg_model = joblib.load("saved_model/fetal_health_logreg_model.pkl")
    logreg_scaler = joblib.load("saved_model/fetal_health_logreg_scaler.pkl")

    X_test_scaled = logreg_scaler.transform(X_test)
    y_pred = logreg_model.predict(X_test_scaled)
    y_prob = logreg_model.predict_proba(X_test_scaled)

    calculate_display_metrics(y_test, y_pred, y_prob, model_name="Logistic Regression")

def decision_tree_prediction(X_test, y_test):
    decision_tree_model = joblib.load("saved_model/decision_tree_model.pkl")
    decision_tree_scaler = joblib.load("saved_model/decision_tree_scaler.pkl")

    X_test_scaled = decision_tree_scaler.transform(X_test)
    y_pred = decision_tree_model.predict(X_test_scaled)
    y_prob = decision_tree_model.predict_proba(X_test_scaled)

    calculate_display_metrics(y_test, y_pred, y_prob, model_name="Decision Tree")

def knn_prediction(X_test, y_test):
    knn_model = joblib.load("saved_model/knn_model.pkl")
    knn_scaler = joblib.load("saved_model/knn_scaler.pkl")

    X_test_scaled = knn_scaler.transform(X_test)
    y_pred = knn_model.predict(X_test_scaled)
    y_prob = knn_model.predict_proba(X_test_scaled)

    calculate_display_metrics(y_test, y_pred, y_prob, model_name="K-Nearest Neighbors")

def naive_bayes_prediction(X_test, y_test):
    naive_bayes_model = joblib.load("saved_model/naive_bayes_model.pkl")
    naive_bayes_scaler = joblib.load("saved_model/naive_bayes_scaler.pkl")

    X_test_scaled = naive_bayes_scaler.transform(X_test)
    y_pred = naive_bayes_model.predict(X_test_scaled)
    y_prob = naive_bayes_model.predict_proba(X_test_scaled)

    calculate_display_metrics(y_test, y_pred, y_prob, model_name="Naive Bayes Classifier")

def random_forest_prediction(X_test, y_test):
    random_forest_model = joblib.load("saved_model/random_forest_model.pkl")
    random_forest_scaler = joblib.load("saved_model/random_forest_scaler.pkl")

    X_test_scaled = random_forest_scaler.transform(X_test)
    y_pred = random_forest_model.predict(X_test_scaled)
    y_prob = random_forest_model.predict_proba(X_test_scaled)

    calculate_display_metrics(y_test, y_pred, y_prob, model_name="Random Forest Classifier")

st.set_page_config(layout="wide")
st.title("Fetal Health Classification Dashboard")

uploaded_file = st.file_uploader(
    "Upload (.csv)",
    type="csv",
    accept_multiple_files=False,
    label_visibility="visible",
    width=500,
)

selected_models = st.multiselect(
    "Select the model(s) you want to use for Fetal Health Classification:",
    model_list,
    default=None,
    width=500,
)

evaluate_clicked = st.button("Train and Evaluate Model(s)", type="primary")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    X_test = test_data.drop("fetal_health", axis=1)
    y_test = test_data["fetal_health"]

    if evaluate_clicked:
        if not selected_models:
            st.warning("Please select at least one model to evaluate.")

        if "Logistic Regression" in selected_models:
            logistic_regression_prediction(X_test, y_test)

        if "Decision Tree" in selected_models:
            decision_tree_prediction(X_test, y_test)

        if "K-Nearest Neighbor Classifier" in selected_models:
            knn_prediction(X_test, y_test)

        if "Naive Bayes Classifier" in selected_models:
            naive_bayes_prediction(X_test, y_test)

        if "Ensemble Model - Random Forest" in selected_models:
            random_forest_prediction(X_test, y_test)
else:
    st.info("Awaiting CSV file upload to perform evaluation.")
