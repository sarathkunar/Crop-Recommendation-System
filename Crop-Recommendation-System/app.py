from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load files
model = pickle.load(open("crop_model.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))

crop_encoder = pickle.load(open("crop_encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:

        N = float(request.form["N"])

        P = float(request.form["P"])

        K = float(request.form["K"])

        temperature = float(request.form["temperature"])

        humidity = float(request.form["humidity"])

        ph = float(request.form["ph"])

        rainfall = float(request.form["rainfall"])


        # Create array
        data = np.array([
            [N, P, K, temperature, humidity, ph, rainfall,]
        ])

        # Scale
        scaled_data = scaler.transform(data)

        # Predict
        prediction = model.predict(scaled_data)

        # Decode crop
        crop = crop_encoder.inverse_transform(prediction)[0]

        return render_template(
            "index.html",
            prediction_text=f"Recommended Crop: {crop}"
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)