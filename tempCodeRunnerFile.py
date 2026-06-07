from flask import Flask, render_template, request, redirect, url_for, send_file
import pickle, os, numpy as np, csv
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from report import generate_pdf

# Visualization helpers
from visualize import (
    plot_prediction_probs,
    generate_all_training_graphs
)

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
GRAPHS_FOLDER = os.path.join("static", "graphs")
DATASET_DIR   = os.path.join("dataset", "images")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPHS_FOLDER, exist_ok=True)

# ── Load models ───────────────────────────────────────────────────────────────
efficientnet_model = load_model("models/efficientnet_model.h5")
text_model         = pickle.load(open("models/text_model.pkl", "rb"))
scaler             = pickle.load(open("models/scaler.pkl",     "rb"))

CLASS_NAMES = ["Benign", "Malignant", "Normal"]

# ── Load saved training accuracies safely ─────────────────────────────────────
try:
    with open("models/accuracy.json") as f:
        _acc = json.load(f)
    TRAIN_ACCURACY = _acc.get("train_accuracy",    "N/A")
    VAL_ACCURACY   = _acc.get("val_accuracy",      "N/A")   # TTA result (most accurate)
    VAL_ACCURACY_RAW = _acc.get("val_accuracy_raw","N/A")   # ADDED: raw (non-TTA)
except (FileNotFoundError, json.JSONDecodeError):
    TRAIN_ACCURACY   = "N/A"
    VAL_ACCURACY     = "N/A"
    VAL_ACCURACY_RAW = "N/A"

# ADDED: Load optimal classification threshold for cancer prediction
try:
    _thresh_data      = pickle.load(open("models/threshold.pkl", "rb"))
    CANCER_THRESHOLD  = _thresh_data.get("threshold", 0.5)
    CANCER_CLASS_IDX  = _thresh_data.get("cancer_class_idx", 1)
except (FileNotFoundError, KeyError):
    CANCER_THRESHOLD = 0.5
    CANCER_CLASS_IDX = 1
    print("⚠ threshold.pkl not found — using default 0.5")

print(f"✅ Cancer classification threshold: {CANCER_THRESHOLD:.3f}")

GRAPH_MANIFEST = {
    "accuracy"    : "graphs/accuracy.png",
    "loss"        : "graphs/loss.png",
    "conf_matrix" : "graphs/confusion_matrix.png",
    "class_dist"  : "graphs/class_distribution.png",
    "pred_probs"  : "graphs/prediction_probs.png",
    "roc"         : "graphs/roc_curve.png",
}

def available_graphs():
    return {k: v for k, v in GRAPH_MANIFEST.items()
            if os.path.isfile(os.path.join("static", v))}

# ── Feature engineering helper (matches train_clinical.py exactly) ────────────
# Uses the same FIXED constants (S_MIN=1, S_MAX=10) — no data leakage.
# OPTIMIZED: Added polynomial features to match v2 of train_clinical.py
def engineer_clinical_features(age_v, smk_v, alc_v, gen_v):
    """
    Builds the same feature vector that train_clinical.py v2 produces.
    Uses fixed constants — NOT dataset statistics — for zero leakage risk.

    Features (in order):
      0  age               — original
      1  smoking           — original
      2  alcohol           — original
      3  genetic_risk      — original
      4  age_group         — derived [0, 1, 2]
      5  smoking_risk      — derived (quadratic scaled)
      6  age_x_smoking     — interaction
      7  smoking_x_alcohol — interaction
      8  total_risk        — composite (smoking + alcohol + genetic)
      9  smoking_sq        — ADDED: polynomial
      10 age_sq            — ADDED: polynomial
    """
    S_MIN, S_MAX = 1, 10   # FIXED constants — same as train_clinical.py

    age_group         = 0 if age_v <= 35 else (1 if age_v <= 55 else 2)
    smoking_risk      = ((min(max(smk_v, S_MIN), S_MAX) - S_MIN) / (S_MAX - S_MIN)) ** 2
    age_x_smoking     = age_v * smk_v
    smoking_x_alcohol = smk_v * alc_v
    total_risk        = smk_v + alc_v + gen_v
    smoking_sq        = smk_v ** 2    # ADDED: matches train_clinical.py v2
    age_sq            = age_v ** 2    # ADDED: matches train_clinical.py v2

    return np.array([[
        age_v, smk_v, alc_v, gen_v,   # original 4
        age_group,                     # derived
        smoking_risk,                  # derived
        age_x_smoking,                 # interaction
        smoking_x_alcohol,             # interaction
        total_risk,                    # composite
        smoking_sq,                    # ADDED: polynomial
        age_sq                         # ADDED: polynomial
    ]])

# ── Save prediction history ────────────────────────────────────────────────────
def save_prediction(data):
    with open('prediction_history.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# ADDED: Risk score calculator for patient-facing output
def compute_risk_score(image_probs, clinical_proba, img_class_idx):
    """
    Computes a 0–100 risk score by fusing image and clinical probabilities.
    Image model: Malignant + Benign (non-normal) classes count as risk.
    Clinical: probability of cancer class.
    """
    # Image risk: complement of "Normal" class probability
    normal_idx    = 2   # CLASS_NAMES = ["Benign", "Malignant", "Normal"]
    image_risk    = 1.0 - float(image_probs[normal_idx])

    # Clinical risk: probability of cancer class
    clinical_risk = float(clinical_proba)

    # Weighted fusion: image model gets slightly more weight (it sees the scan)
    fused_risk = 0.55 * image_risk + 0.45 * clinical_risk
    return round(fused_risk * 100, 1)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_file = request.files.get("image")
        if not img_file or img_file.filename == "":
            return "No image uploaded.", 400

        filename      = secure_filename(img_file.filename)
        img_save_path = os.path.join(UPLOAD_FOLDER, filename)
        img_file.save(img_save_path)

        # ── EfficientNet Image Prediction ────────────────────────────────────
        img     = keras_image.load_img(img_save_path, target_size=(224, 224))
        img_arr = keras_image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        img_pred      = efficientnet_model.predict(img_arr)
        probs         = img_pred[0]
        img_class_idx = int(np.argmax(probs))
        image_label   = CLASS_NAMES[img_class_idx]
        confidence    = round(float(probs[img_class_idx]) * 100, 2)

        plot_prediction_probs(probs, CLASS_NAMES)

        # ── Clinical Data ─────────────────────────────────────────────────────
        age          = int(request.form["age"])
        smoking      = int(request.form["smoking"])
        alcohol      = int(request.form["alcohol"])
        genetic_risk = int(request.form["genetic_risk"])

        # ── Feature engineering + scaling ─────────────────────────────────────
        clinical_arr = engineer_clinical_features(age, smoking, alcohol, genetic_risk)
        clinical_arr = scaler.transform(clinical_arr)

        # OPTIMIZED: Use calibrated probability + threshold for cancer prediction
        # instead of raw predict() with default 0.5 threshold
        if hasattr(text_model, "predict_proba"):
            clinical_proba_all  = text_model.predict_proba(clinical_arr)[0]
            clinical_prob_cancer = float(clinical_proba_all[CANCER_CLASS_IDX])
            clinical_class      = int(clinical_prob_cancer >= CANCER_THRESHOLD)
        else:
            # Fallback for models without predict_proba
            clinical_class       = int(text_model.predict(clinical_arr)[0])
            clinical_prob_cancer = float(clinical_class)

        clinical_result = "Cancer" if clinical_class == CANCER_CLASS_IDX else "Normal"

        # ── Hybrid Final Result ───────────────────────────────────────────────
        image_result = "Cancer" if image_label != "Normal" else "Normal"

        if image_result == "Cancer" and clinical_result == "Cancer":
            final_result = "High Risk"
        elif image_result == "Cancer" or clinical_result == "Cancer":
            final_result = "Medium Risk"
        else:
            final_result = "Low Risk"

        # ADDED: Fused risk score (0–100)
        risk_score = compute_risk_score(probs, clinical_prob_cancer, img_class_idx)

        # ── Save history ──────────────────────────────────────────────────────
        save_prediction([
            datetime.now(),
            age,
            smoking,
            alcohol,
            genetic_risk,
            image_label,
            clinical_result,
            final_result,
            confidence,
            risk_score       # ADDED
        ])

        # ── Generate PDF ──────────────────────────────────────────────────────
        patient_data = {
            "Age"                : age,
            "Smoking"            : smoking,
            "Alcohol"            : alcohol,
            "Genetic Risk"       : genetic_risk,
            "Image Prediction"   : image_label,
            "Clinical Prediction": clinical_result,
            "Final Risk"         : final_result,
            "Confidence"         : confidence,
            "Risk Score"         : risk_score,    # ADDED
        }

        pdf_path = os.path.join("static", "report.pdf")
        generate_pdf(pdf_path, patient_data)

        return render_template("result.html",
            image_path          = "uploads/" + filename,
            image_prediction    = image_label,
            clinical_prediction = clinical_result,
            final_result        = final_result,
            confidence          = confidence,
            risk_score          = risk_score,          # ADDED
            age                 = age,
            smoking             = smoking,
            alcohol             = alcohol,
            genetic_risk        = genetic_risk,
            graphs              = available_graphs(),
            train_accuracy      = TRAIN_ACCURACY,
            val_accuracy        = VAL_ACCURACY,
            val_accuracy_raw    = VAL_ACCURACY_RAW,    # ADDED
        )

    except Exception as e:
        import traceback
        return f"<h3>Error: {e}</h3><pre>{traceback.format_exc()}</pre>"   # OPTIMIZED: full traceback


@app.route("/download_report")
def download_report():
    return send_file("static/report.pdf", as_attachment=True)


@app.route("/graphs")
def graphs_page():
    labels = {
        "accuracy"    : "EfficientNet Training Accuracy",
        "loss"        : "EfficientNet Training Loss",
        "conf_matrix" : "Confusion Matrix",
        "class_dist"  : "Class Distribution",
        "pred_probs"  : "Prediction Probabilities",
        "roc"         : "ROC Curve",
    }
    graphs = {labels[k]: v for k, v in available_graphs().items()}
    return render_template("graphs.html", graphs=graphs)


if __name__ == "__main__":
    app.run(debug=True)