"""
Feature Analysis and Exploratory Data Preprocessing Module
Performs statistical analysis, correlation assessment, and feature importance
evaluation on the crop recommendation dataset.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")

def run_feature_analysis():
    print("=" * 60)
    print("🌾 CROP RECOMMENDATION SYSTEM - FEATURE ANALYSIS PIPELINE")
    print("=" * 60)

    # 1. Load Dataset
    df = pd.read_csv(DATASET_PATH)
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    print(f"\n1. DATASET OVERVIEW:")
    print(f"   - Total records: {df.shape[0]} (deduplicated from {initial_shape[0]})")
    print(f"   - Target classes (crops): {df['label'].nunique()}")
    print(f"   - Features: {list(df.columns[:-1])}")

    # 2. Statistical Summary
    stats = df.describe().to_dict()

    # 3. Label Encoding for Categorical Attributes
    le_season = LabelEncoder()
    df["season_encoded"] = le_season.fit_transform(df["season"])
    le_crop = LabelEncoder()
    df["label_encoded"] = le_crop.fit_transform(df["label"])

    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "season_encoded"]
    feature_labels = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature", "Humidity", "pH", "Rainfall", "Season"]
    X = df[feature_cols]
    y = df["label_encoded"]

    # 4. Correlation Analysis
    corr_matrix = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "season_encoded"]].corr()

    # 5. Mutual Information Scores
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # 6. ANOVA F-Score
    f_scores, p_values = f_classif(X, y)

    # 7. Random Forest Feature Importance (MDI)
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=3, random_state=42)
    rf.fit(X, y)
    rf_importances = rf.feature_importances_

    # Aggregate into summary table
    feature_df = pd.DataFrame({
        "Feature": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "season"],
        "Display_Name": feature_labels,
        "RF_Importance_%": np.round(rf_importances * 100, 2),
        "Mutual_Info": np.round(mi_scores, 4),
        "ANOVA_F_Score": np.round(f_scores, 2),
    }).sort_values(by="RF_Importance_%", ascending=False).reset_index(drop=True)

    print("\n2. FEATURE IMPORTANCE RANKING (Random Forest + Information Gain):")
    print(feature_df.to_string(index=False))

    # 8. Export Reports
    report = {
        "dataset_summary": {
            "total_samples": int(df.shape[0]),
            "feature_count": len(feature_cols),
            "target_classes": int(df['label'].nunique()),
            "classes": sorted(df['label'].unique().tolist())
        },
        "feature_rankings": feature_df.to_dict(orient="records"),
        "correlation_with_nutrients": {
            col: corr_matrix[col].to_dict() for col in ["N", "P", "K"]
        }
    }

    report_path = os.path.join(BASE_DIR, "feature_analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved feature analysis report: {report_path}")

    csv_path = os.path.join(BASE_DIR, "feature_importance.csv")
    feature_df.to_csv(csv_path, index=False)
    print(f"Saved feature rankings CSV: {csv_path}")

    return report

if __name__ == "__main__":
    run_feature_analysis()
