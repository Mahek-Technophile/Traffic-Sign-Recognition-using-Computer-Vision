from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "model.keras"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_FOLDER", BASE_DIR / "uploads"))
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMAGE_SIZE = (32, 32)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

CLASS_NAMES = [
    "Speed Limit 20 km/h", "Speed Limit 30 km/h", "Speed Limit 50 km/h",
    "Speed Limit 60 km/h", "Speed Limit 70 km/h", "Speed Limit 80 km/h",
    "End of Speed Limit 80 km/h", "Speed Limit 100 km/h", "Speed Limit 120 km/h",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]


def preprocessing(img: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to the model's normalized grayscale input."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("The uploaded file is not a valid image.")
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unsupported image format.")
    gray = cv2.resize(gray, IMAGE_SIZE)
    gray = cv2.equalizeHist(gray)
    return gray.astype(np.float32) / 255.0


@lru_cache(maxsize=1)
def load_trained_model():
    """Load lazily so importing the Flask app does not fail before first use."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run main.py to create model.keras."
        )
    from tensorflow.keras.models import load_model

    return load_model(str(MODEL_PATH))


def model_predict(img_path: str | os.PathLike, model=None) -> str:
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")
    batch = preprocessing(img).reshape(1, IMAGE_SIZE[1], IMAGE_SIZE[0], 1)
    predictor = model if model is not None else load_trained_model()
    predictions = np.asarray(predictor.predict(batch, verbose=0))
    if predictions.ndim != 2 or predictions.shape[0] != 1 or predictions.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Model must return one prediction across 43 classes; got {predictions.shape}."
        )
    return CLASS_NAMES[int(np.argmax(predictions[0]))]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def upload():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="Please choose an image to upload."), 400
    if not allowed_file(uploaded.filename):
        return jsonify(error="Unsupported file type. Upload a PNG, JPG, or JPEG image."), 400

    filename = secure_filename(uploaded.filename)
    if not filename:
        return jsonify(error="Invalid filename."), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / filename
    try:
        uploaded.save(file_path)
        prediction = model_predict(file_path)
        return prediction, 200
    except FileNotFoundError as exc:
        app.logger.exception("Model file is missing")
        return jsonify(error=str(exc)), 503
    except (ValueError, OSError) as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction failed. Check the model and server logs."), 500
    finally:
        file_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
