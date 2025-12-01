import os
from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "../models/knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")

# Load model + scaler
model = load(MODEL_PATH)
scaler = load(SCALER_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get input data
    f1 = float(request.form["f1"])
    f2 = float(request.form["f2"])
    f3 = float(request.form["f3"])
    f4 = float(request.form["f4"])

    # Prepare array
    input_data = np.array([[f1, f2, f3, f4]])

    # Scale
    scaled_input = scaler.transform(input_data)

    # Predict
    result = model.predict(scaled_input)[0]

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)