import joblib
import numpy as np
import os

root = os.path.dirname(__file__)

model_bundle = joblib.load(os.path.join(root, 'heart_disease_model.pkl'))
scaler = joblib.load(os.path.join(root, 'heart_scaler.pkl'))

model = model_bundle['model']
features = model_bundle['feature_names']

print('Feature names:', features)

# Sample input
sample = np.array([[50, 1, 120, 80, 24.0, 1, 1, 0, 0, 1]])

scaled = scaler.transform(sample)

print('Prediction:', model.predict(scaled))
print('Probabilities:', model.predict_proba(scaled))