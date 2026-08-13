#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self):
        self.data = None
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.y_train = None
        self.y_val = None
        self.scaler = None
        self.preprocess_data()

    def load_data(self):
        # Load dataset from the specified CSV file.
        self.data = pd.read_csv("fetal_health.csv")
        print(self.data.info())
        print(self.data.isnull().sum())

    def plot_correlation_heatmap(self):
        #Plot a heatmap to visualize feature correlations.
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.data.corr(numeric_only=True)
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Feature Correlation Heatmap')
        plt.show()

    def preprocess_data(self):
        # Split the dataset into features and target variable, then standardize the features.
        self.load_data()

        X = self.data.drop('fetal_health', axis=1)
        y = self.data['fetal_health']

        # Split the dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Standardize the features using z-score normalization
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)  # Scale the training data
        self.X_val_scaled = scaler.transform(X_val)  # Scale the validation data
        self.y_train = y_train
        self.y_val = y_val
        self.scaler = scaler
