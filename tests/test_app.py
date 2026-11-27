import unittest
import json
from app import app, VALID_SEASONS

class CropRecommendationSystemTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        """Test GET / returns 200 and loads HTML form."""
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Smart Crop Recommendation System", response.data)
        self.assertIn(b"Select Season", response.data)
        self.assertIn(b"Feature Importance Analysis", response.data)

    def test_health_check(self):
        """Test GET /health returns healthy status."""
        response = self.app.get("/health")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["model_type"], "RandomForestClassifier")
        self.assertEqual(data["available_seasons"], VALID_SEASONS)

    def test_model_info_endpoint(self):
        """Test GET /api/model-info returns metadata and benchmarks."""
        response = self.app.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("model", data)
        self.assertIn("feature_columns", data)

    def test_predict_form_success(self):
        """Test valid form POST to /predict returns recommendation and confidence."""
        payload = {
            "N": "90",
            "P": "42",
            "K": "43",
            "temperature": "20.8",
            "humidity": "82.0",
            "ph": "6.5",
            "rainfall": "202.9",
            "season": "Summer"
        }
        response = self.app.post("/predict", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Recommended Crop:", response.data)
        self.assertIn(b"Confidence:", response.data)

    def test_predict_form_validation_error(self):
        """Test invalid form POST returns 400 with helpful error."""
        payload = {
            "N": "90",
            "P": "42",
            "K": "43",
            "temperature": "20.8",
            "humidity": "82.0",
            "ph": "99",  # Invalid pH > 14
            "rainfall": "202.9",
            "season": "Summer"
        }
        response = self.app.post("/predict", data=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"pH must be between 0.0 and 14.0", response.data)

    def test_predict_form_missing_field(self):
        """Test form POST with missing field returns 400."""
        payload = {
            "N": "90",
            "P": "42",
            "K": "43",
            "temperature": "20.8",
            "humidity": "82.0",
            "ph": "6.5",
            "rainfall": "202.9"
            # Missing season
        }
        response = self.app.post("/predict", data=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Season", response.data)
        self.assertIn(b"is required", response.data)

    def test_api_predict_json_success(self):
        """Test JSON POST to /api/predict returns structured JSON with probabilities."""
        payload = {
            "N": 65,
            "P": 29,
            "K": 45,
            "temperature": 14.2,
            "humidity": 54.6,
            "ph": 6.6,
            "rainfall": 70.5,
            "season": "Winter"
        }
        response = self.app.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("recommended_crop", data)
        self.assertIn("confidence_percent", data)
        self.assertIn("top_recommendations", data)
        self.assertIsInstance(data["top_recommendations"], list)
        self.assertGreater(len(data["top_recommendations"]), 0)

    def test_api_predict_invalid_season(self):
        """Test API POST with invalid season returns 400 error JSON."""
        payload = {
            "N": 65,
            "P": 29,
            "K": 45,
            "temperature": 14.2,
            "humidity": 54.6,
            "ph": 6.6,
            "rainfall": 70.5,
            "season": "Monsoon"  # Invalid season
        }
        response = self.app.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "error")
        self.assertIn("Invalid season", data["message"])

if __name__ == "__main__":
    unittest.main()
