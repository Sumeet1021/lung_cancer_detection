"""
app.py — Clean Flask App
=========================
Routes:
  GET  /         -> upload page  (templates/index.html)
  POST /predict  -> result page  (templates/result.html)
  GET  /graphs   -> graphs page  (templates/graphs.html)

FIX: use tf_keras to load model — matches how train_efficientnet.py saved it.
     standalone keras (Keras 3) cannot load tf_keras-saved .h5 files.
"""

import os, pickle, json, csv
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF spam

import numpy as np
from datetime import datetime
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# ── MUST match train_efficientnet.py: use tf_keras, not keras/tensorflow.keras ─
import tf_keras
from tf_keras.models import load_model
from tf_keras.preprocessing import image as keras_image

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
print("Loading models...")
efficientnet_model = load_model("models/efficientnet_model.h5", compile=False)
clinical_model     = pickle.load(open("models/clinical_model.pkl", "rb"))
scaler             = pickle.load(open("models/scaler.pkl",         "rb"))
label_encoder      = pickle.load(open("models/label_encoder.pkl", "rb"))
print("All models loaded.")

# ── CLASS NAMES ───────────────────────────────────────────────────────────────
if os.path.isdir("dataset/images"):
    CLASS_NAMES = sorted([
        d for d in os.listdir("dataset/images")
        if os.path.isdir(os.path.join("dataset/images", d))
    ])
else:
    CLASS_NAMES = ["bengin", "malignant", "normal"]

print(f"Image classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

model_output_size = efficientnet_model.output_shape[-1]
if model_output_size != len(CLASS_NAMES):
    print(f"WARNING: model outputs {model_output_size} classes, adjusting CLASS_NAMES.")
    CLASS_NAMES = CLASS_NAMES[:model_output_size]

# ── LOAD TRAINING ACCURACY ────────────────────────────────────────────────────
try:
    with open("models/accuracy.json") as f:
        _acc = json.load(f)
    TRAIN_ACC = _acc.get("train_accuracy", "N/A")
    VAL_ACC   = _acc.get("val_accuracy",   "N/A")
except (FileNotFoundError, json.JSONDecodeError):
    TRAIN_ACC = VAL_ACC = "N/A"

# ── FEATURE ENGINEERING (must match train_clinical.py exactly) ────────────────
def engineer_features(age, smoking, alcohol, genetic_risk):
    S_MIN, S_MAX = 1, 10
    age_group         = 0 if age <= 35 else (1 if age <= 55 else 2)
    smoking_risk      = ((min(max(smoking, S_MIN), S_MAX) - S_MIN) / (S_MAX - S_MIN)) ** 2
    age_x_smoking     = age * smoking
    smoking_x_alcohol = smoking * alcohol
    total_risk        = smoking + alcohol + genetic_risk
    return np.array([[
        age, smoking, alcohol, genetic_risk,
        age_group, smoking_risk,
        age_x_smoking, smoking_x_alcohol,
        total_risk,
    ]])

# ── RISK FUSION ───────────────────────────────────────────────────────────────
def get_risk_level(image_label, clinical_result):
    img_cancer  = image_label.lower() not in {"normal"}
    clin_cancer = clinical_result == "Cancer"
    if img_cancer and clin_cancer:
        return "High Risk"
    elif img_cancer or clin_cancer:
        return "Medium Risk"
    return "Low Risk"

def compute_risk_score(image_probs, cancer_prob):
    normal_idx = next(
        (i for i, c in enumerate(CLASS_NAMES) if c.strip().lower() == "normal"), -1
    )
    if 0 <= normal_idx < len(image_probs):
        image_risk = 1.0 - float(image_probs[normal_idx])
    else:
        image_risk = 1.0 - float(np.max(image_probs))
    return round((0.55 * image_risk + 0.45 * float(cancer_prob)) * 100, 1)

# ── PREPROCESSING: must match training (preprocess_input, not /255) ───────────
from tf_keras.applications.efficientnet import preprocess_input

def prepare_image(img_path):
    img     = keras_image.load_img(img_path, target_size=(224, 224))
    img_arr = keras_image.img_to_array(img)          # float32, 0-255
    img_arr = preprocess_input(img_arr)              # EfficientNet normalization
    return np.expand_dims(img_arr, axis=0)

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_file = request.files.get("image")
        if not img_file or img_file.filename == "":
            return "No image uploaded.", 400

        filename  = secure_filename(img_file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        img_file.save(save_path)

        # ── Image prediction ──────────────────────────────────────────────────
        img_arr = prepare_image(save_path)
        probs   = efficientnet_model.predict(img_arr, verbose=0)[0]

        effective   = min(len(probs), len(CLASS_NAMES))
        img_cls_idx = int(np.argmax(probs[:effective]))
        image_label = CLASS_NAMES[img_cls_idx]
        confidence  = round(float(probs[img_cls_idx]) * 100, 2)

        # ── Clinical prediction ───────────────────────────────────────────────
        age          = int(request.form["age"])
        smoking      = int(request.form["smoking"])
        alcohol      = int(request.form["alcohol"])
        genetic_risk = int(request.form["genetic_risk"])

        feat_arr = scaler.transform(engineer_features(age, smoking, alcohol, genetic_risk))

        if hasattr(clinical_model, "predict_proba"):
            proba_all  = clinical_model.predict_proba(feat_arr)[0]
            classes    = list(label_encoder.classes_)
            cancer_idx = next(
                (i for i, c in enumerate(classes)
                 if str(c).strip().lower() in {"yes", "cancer", "1", "positive"}),
                min(1, len(classes) - 1)
            )
            cancer_prob = float(proba_all[cancer_idx])
        else:
            cancer_prob = float(clinical_model.predict(feat_arr)[0])

        raw_label       = str(label_encoder.inverse_transform(
                              [int(clinical_model.predict(feat_arr)[0])])[0])
        is_cancer       = raw_label.strip().lower() in {"yes", "cancer", "1", "positive"}
        clinical_result = "Cancer" if is_cancer else "Normal"

        # ── Fusion ────────────────────────────────────────────────────────────
        final_result = get_risk_level(image_label, clinical_result)
        risk_score   = compute_risk_score(probs, cancer_prob)

        # ── Log ───────────────────────────────────────────────────────────────
        with open("prediction_history.csv", "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(), age, smoking, alcohol, genetic_risk,
                image_label, clinical_result, final_result, confidence, risk_score
            ])

        return render_template("result.html",
            image_path          = "uploads/" + filename,
            image_prediction    = image_label,
            confidence          = confidence,
            clinical_prediction = clinical_result,
            final_result        = final_result,
            risk_score          = risk_score,
            age                 = age,
            smoking             = smoking,
            alcohol             = alcohol,
            genetic_risk        = genetic_risk,
            train_accuracy      = TRAIN_ACC,
            val_accuracy        = VAL_ACC,
        )

    except Exception as e:
        import traceback
        return f"<h3>Error: {e}</h3><pre>{traceback.format_exc()}</pre>", 500


# ── GRAPHS ROUTE ──────────────────────────────────────────────────────────────
@app.route("/graphs")
def graphs():
    graph_files = {
        "Training Accuracy":        "graphs/accuracy.png",
        "Training Loss":            "graphs/loss.png",
        "Confusion Matrix":         "graphs/confusion_matrix.png",
        "Class Distribution":       "graphs/class_distribution.png",
        "Prediction Probabilities": "graphs/prediction_probs.png",
        "ROC Curve":                "graphs/roc_curve.png",
    }

    # Only pass graphs that actually exist on disk
    available_graphs = {
        name: path
        for name, path in graph_files.items()
        if os.path.exists(os.path.join("static", path))
    }

    return render_template("graphs.html", graphs=available_graphs)


if __name__ == "__main__":
    app.run(debug=True)