import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    log_loss,
    ConfusionMatrixDisplay
)

class LogisticRegressionModel:
    def __init__(self, X_train_scaled, X_val_scaled, y_train, y_val):
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.y_train = y_train
        self.y_val = y_val
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        

    def train(self):
        self.model.fit(self.X_train_scaled, self.y_train)
        self.evaluate(self.X_val_scaled, self.y_val)

    def predict(self, X_val):
        return self.model.predict(X_val)

    def predict_proba(self, X_val):
        return self.model.predict_proba(X_val)

    def evaluate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        y_pred_proba = self.predict_proba(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred)
        class_report = classification_report(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
        logloss = log_loss(y_val, y_pred_proba)

        print(f'Accuracy: {accuracy}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        print(f'Classification Report:\n{class_report}')
        print(f'ROC AUC Score: {roc_auc}')
        print(f'Log Loss: {logloss}')

        # display the confusion matrix
        ConfusionMatrixDisplay(conf_matrix).plot()

