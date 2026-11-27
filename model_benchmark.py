"""
Model Benchmark and Algorithm Evaluation Module
Evaluates multiple classification algorithms using cross-validation and standard ML metrics.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")

def evaluate_algorithms():
    print("=" * 60)
    print("🌾 CROP RECOMMENDATION SYSTEM - MODEL BENCHMARK PIPELINE")
    print("=" * 60)

    # 1. Load and Preprocess
    df = pd.read_csv(DATASET_PATH).drop_duplicates()
    le_season = LabelEncoder()
    df["season"] = le_season.fit_transform(df["season"])
    le_crop = LabelEncoder()
    df["label"] = le_crop.fit_transform(df["label"])

    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "season"]
    X = df[feature_cols]
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Candidate Classification Algorithms
    models = {
        "Tuned Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=3, random_state=42
        ),
        "Standard Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        "Support Vector Machine (RBF)": SVC(
            kernel="rbf", probability=True, random_state=42
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        )
    }

    # 3. Stratified 5-Fold Cross Validation & Holdout Evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("\nTraining and evaluating models across 5 folds...")
    for name, clf in models.items():
        # Cross validation scores
        cv_res = cross_validate(clf, X_scaled, y, cv=cv, scoring=["accuracy", "f1_weighted"])
        cv_acc_mean = float(cv_res["test_accuracy"].mean())
        cv_acc_std = float(cv_res["test_accuracy"].std())
        cv_f1_mean = float(cv_res["test_f1_weighted"].mean())

        # Fit on train and evaluate on holdout test set
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        test_acc = float(accuracy_score(y_test, y_pred))
        test_precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        test_recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        test_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        results.append({
            "Algorithm": name,
            "CV_Accuracy_%": round(cv_acc_mean * 100, 2),
            "CV_Std_%": round(cv_acc_std * 100, 2),
            "CV_F1_%": round(cv_f1_mean * 100, 2),
            "Test_Accuracy_%": round(test_acc * 100, 2),
            "Test_Precision_%": round(test_precision * 100, 2),
            "Test_Recall_%": round(test_recall * 100, 2),
            "Test_F1_%": round(test_f1 * 100, 2),
        })

    # Sort results by Test F1
    results_df = pd.DataFrame(results).sort_values(by="CV_Accuracy_%", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("MODEL BENCHMARK PERFORMANCE COMPARISON")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Save CSV and JSON reports
    csv_path = os.path.join(BASE_DIR, "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark CSV: {csv_path}")

    json_path = os.path.join(BASE_DIR, "model_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark JSON: {json_path}")

    best_model_name = results_df.iloc[0]["Algorithm"]
    best_acc = results_df.iloc[0]["Test_Accuracy_%"]
    print(f"\n🏆 Best Performing Classifier: {best_model_name} (Test Accuracy: {best_acc}%)")

    return results

if __name__ == "__main__":
    evaluate_algorithms()
