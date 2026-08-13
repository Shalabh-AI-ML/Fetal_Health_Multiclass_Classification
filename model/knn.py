import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path
# Add parent directory to path to import data_processing
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing import DataProcessor
from sklearn.neighbors import KNeighborsClassifier

# initialize the data processor
data_processor = DataProcessor()

X_train_scaled = data_processor.X_train_scaled
X_val_scaled = data_processor.X_val_scaled
y_train = data_processor.y_train
y_val = data_processor.y_val

print("Training K-Nearest Neighbors Model...")
knn_model = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)
knn_model.fit(X_train_scaled, y_train)
print("K-Nearest Neighbors Model Trained.")

# Evaluate the model on the validation set
val_accuracy = knn_model.score(X_val_scaled, y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Save the trained model
joblib.dump(knn_model, "saved_model/knn_model.pkl")
print("Model saved successfully.")

joblib.dump(data_processor.scaler, "saved_model/knn_scaler.pkl")
print("Scaler saved successfully.")