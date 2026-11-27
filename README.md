# 🌾 Smart Crop Recommendation System

An AI-driven agricultural decision-support platform that predicts the optimal crop to cultivate based on soil nutrient levels, weather parameters, and growing season.

---

## 🎯 Project Highlights

- **Machine Learning Inference**: Evaluated and benchmarked **7 classification algorithms** using 5-Fold Stratified Cross-Validation and holdout testing. The tuned **Random Forest Classifier** achieved the highest overall performance (**78.00% Test Accuracy**, **75.67% CV Accuracy**).
- **Feature Analysis & Importance**: Quantified the influence of each agricultural parameter using Mean Decrease in Impurity (MDI), Mutual Information (MI), and ANOVA F-tests.
- **Probabilistic Multi-Crop Recommendation**: Predicts the best crop along with confidence percentages and alternative viable crops.
- **Modern Responsive Web UI**: Glassmorphic dashboard with live validation, client-side input safety limits, and loading feedback.
- **REST API Support**: Dedicated JSON endpoints (`/api/predict` and `/api/model-info`) for seamless integration with IoT field sensors, external apps, or microservices.
- **Production-Ready**: Configured for containerization (`Dockerfile`) and WSGI production serving (`gunicorn` via `Procfile`).
- **Automated Test Suite**: Unit and integration test coverage for web forms and REST endpoints.

---

## 📊 Classification Algorithm Benchmark

All models were evaluated on the 1,500-sample dataset using 5-Fold Stratified Cross-Validation and a 20% holdout test set:

| Algorithm | CV Accuracy (%) | CV Std (%) | Test Accuracy (%) | Test Precision (%) | Test Recall (%) | Test F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Tuned Random Forest** | **75.67** | **1.99** | **78.00** | **78.13** | **78.00** | **77.65** |
| Standard Random Forest | 74.87 | 1.69 | 79.00 | 79.61 | 79.00 | 78.98 |
| Extra Trees | 74.20 | 1.07 | 76.67 | 76.96 | 76.67 | 76.47 |
| Gradient Boosting | 72.27 | 2.69 | 75.00 | 75.02 | 75.00 | 74.74 |
| Gaussian Naive Bayes | 72.00 | 1.37 | 72.00 | 73.42 | 72.00 | 71.78 |
| Support Vector Machine (RBF) | 71.60 | 1.02 | 72.33 | 72.80 | 72.33 | 71.97 |
| Logistic Regression | 70.07 | 1.87 | 71.33 | 72.19 | 71.33 | 71.26 |
| Decision Tree | 69.80 | 2.45 | 69.33 | 70.73 | 69.33 | 69.47 |
| K-Nearest Neighbors (KNN) | 64.73 | 2.07 | 66.67 | 67.06 | 66.67 | 66.48 |

---

## 🔍 Feature Analysis & Relative Importance

Relative contribution of each soil nutrient and environmental parameter toward recommendation decisions:

| Feature | Description | RF Importance (%) | Mutual Information | ANOVA F-Score |
| :--- | :--- | :---: | :---: | :---: |
| **season** | Growing Season | **18.60%** | 1.1961 | $\infty$ |
| **P** | Phosphorus level | **14.48%** | 0.6034 | 143.80 |
| **temperature** | Ambient temperature (°C) | **14.44%** | 0.8253 | 342.58 |
| **rainfall** | Precipitation (mm) | **13.07%** | 0.7192 | 327.05 |
| **N** | Nitrogen level | **12.05%** | 0.7157 | 238.49 |
| **K** | Potassium level | **10.73%** | 0.5897 | 144.17 |
| **humidity** | Relative humidity (%) | **9.89%** | 0.5100 | 143.29 |
| **ph** | Soil pH value | **6.74%** | 0.1556 | 19.24 |

---

## 📁 Project Structure

```
Crop-Recommendation-System/
├── Crop_recommendation.csv      # Agricultural dataset (1,500 records)
├── feature_analysis.py          # Feature analysis & statistical preprocessing script
├── model_benchmark.py           # Multi-model classification benchmark pipeline
├── train_model.py               # Model training and artifact generation pipeline
├── app.py                       # Production Flask app (Web UI + REST API)
├── crop_model.pkl               # Tuned RandomForest model
├── scaler.pkl                   # StandardScaler for input feature vectors
├── crop_encoder.pkl             # LabelEncoder for target crop classes (15 crops)
├── season_encoder.pkl           # LabelEncoder for seasons
├── feature_analysis_report.json # Exported feature importance report
├── model_comparison.json        # Benchmark metrics across 7 algorithms
├── templates/
│   └── index.html               # Frontend HTML5 template
├── static/
│   ├── style.css                # Custom glassmorphic CSS styling
│   └── script.js                # Form interaction and loading state handler
├── tests/
│   └── test_app.py              # Automated test suite (8 tests)
├── Dockerfile                   # Docker container specification
├── .dockerignore                # Container build exclusions
├── Procfile                     # Gunicorn web process definition for cloud hosts
├── requirements.txt             # Locked Python dependencies
└── runtime.txt                  # Python runtime specification (3.11.9)
```

---

## 🛠️ Local Setup & Installation

### 1. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing & Model Evaluation (Optional)
```bash
python3 feature_analysis.py    # Generates feature analysis reports
python3 model_benchmark.py     # Runs cross-validated benchmark on 7 models
python3 train_model.py         # Retrains and serializes the best model
```

---

## 🏃 Running the Application

### Development Server
```bash
python3 app.py
```
Open [http://localhost:5001](http://localhost:5001) in your browser.

### Production Server (Gunicorn)
```bash
gunicorn --bind 0.0.0.0:5001 --workers 2 --threads 4 app:app
```

---

## 📡 REST API Documentation

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "RandomForestClassifier",
  "available_seasons": ["Autumn", "Spring", "Summer", "Winter"],
  "features": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "season"]
}
```

### 2. Model & Benchmark Info
- **Endpoint**: `GET /api/model-info`
- **Response**: Returns feature importance rankings and multi-algorithm benchmark scores.

### 3. Crop Prediction
- **Endpoint**: `POST /api/predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9,
  "season": "Summer"
}
```
- **Success Response (HTTP 200)**:
```json
{
  "status": "success",
  "recommended_crop": "Rice",
  "confidence_percent": 86.0,
  "top_recommendations": [
    {"crop": "Rice", "confidence_percent": 86.0},
    {"crop": "Banana", "confidence_percent": 5.0},
    {"crop": "Mango", "confidence_percent": 3.0}
  ],
  "inputs": {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9,
    "season": "Summer"
  }
}
```

---

## 🧪 Running Automated Tests

Run the test suite using `unittest`:
```bash
python3 -m unittest discover -s tests -v
```
All 8 test cases pass with full coverage for data ingestion, form validation, error handling, and JSON REST API.
