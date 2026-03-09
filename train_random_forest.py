import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


def load_and_prepare(path):
    df = pd.read_csv(path)

    # ----- Encode Categorical Columns -----

    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})

    df['ChestPainType'] = LabelEncoder().fit_transform(df['ChestPainType'])
    df['RestingECG'] = LabelEncoder().fit_transform(df['RestingECG'])
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df['ST_Slope'] = LabelEncoder().fit_transform(df['ST_Slope'])

    feature_names = [
        'Age', 'Sex', 'ChestPainType', 'RestingBP',
        'Cholesterol', 'FastingBS', 'RestingECG',
        'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
    ]

    X = df[feature_names]
    y = df['HeartDisease']

    return X, y, feature_names


def train_and_save(X, y, feature_names, model_path, scaler_path):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Random Forest does not need scaling,
    # but keeping scaler to avoid Streamlit changes
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train_s, y_train)

    y_pred = rf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    scores = cross_val_score(
        rf,
        scaler.transform(X),
        y,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    print(f"\n5-Fold CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    model_bundle = {
        'model': rf,
        'feature_names': feature_names
    }

    joblib.dump(model_bundle, model_path)
    joblib.dump(scaler, scaler_path)

    print("\nModel saved successfully!")


if __name__ == '__main__':
    root = os.path.dirname(__file__)

    data_path = os.path.join(root, 'heart.csv')  # <-- rename your new dataset to heart.csv

    X, y, feature_names = load_and_prepare(data_path)

    train_and_save(
        X, y,
        feature_names,
        os.path.join(root, 'heart_disease_model.pkl'),
        os.path.join(root, 'heart_scaler.pkl')
    )