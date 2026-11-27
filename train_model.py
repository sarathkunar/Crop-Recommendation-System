import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")

# Explicit feature list to ensure consistent ordering between training and inference
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "season"]
TARGET_COLUMN = "label"

def train():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Encode season column
    season_encoder = LabelEncoder()
    df["season"] = season_encoder.fit_transform(df["season"])

    # Encode crop label
    crop_encoder = LabelEncoder()
    df[TARGET_COLUMN] = crop_encoder.fit_transform(df[TARGET_COLUMN])

    # Features and Target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Test Split (Stratified for balanced class representation)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Optimized Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=3,
        random_state=42
    )

    # Train
    print("Training optimized RandomForest model...")
    model.fit(X_train, y_train)

    # Predict & Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=crop_encoder.classes_))

    # Feature importance summary
    importances = model.feature_importances_
    print("\nFeature Importances:")
    for col, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True):
        print(f"  - {col}: {imp * 100:.2f}%")

    # Save artifacts
    artifacts = {
        "crop_model.pkl": model,
        "scaler.pkl": scaler,
        "season_encoder.pkl": season_encoder,
        "crop_encoder.pkl": crop_encoder,
    }

    for filename, obj in artifacts.items():
        filepath = os.path.join(BASE_DIR, filename)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        print(f"Saved {filename}")

    print("All artifacts saved successfully.")

if __name__ == "__main__":
    train()
