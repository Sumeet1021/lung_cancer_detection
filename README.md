# 🫁 DeepCancerDetect - Lung Cancer Detection Using Deep Learning

An AI-powered Lung Cancer Detection System that uses Deep Learning and Machine Learning techniques to predict lung cancer from both medical images and clinical patient data.

The project combines:

- CNN-based CT Scan Image Classification
- Clinical Data Prediction using Random Forest
- Medical Image Processing
- Automated Cancer Risk Prediction

---

## 🚀 Features

### 🖼 Image-Based Lung Cancer Detection

- CT Scan Image Classification
- Convolutional Neural Network (CNN)
- Multi-class Prediction
- Automatic Feature Extraction
- Deep Learning Pipeline using TensorFlow

### 📊 Clinical Data Prediction

Predicts lung cancer risk using patient attributes such as:

- Age
- Smoking Habit
- Alcohol Consumption
- Genetic Risk

Model Used:

- Random Forest Classifier

### 🤖 AI-Powered Diagnosis Support

Combines:

- Medical Imaging
- Clinical Information
- Machine Learning Models

to assist in early lung cancer detection.

---

# 🏗 System Architecture

```text
                Patient Data
                      │
                      ▼

           Random Forest Classifier
                      │
                      ▼

             Cancer Risk Prediction


                      +


               CT Scan Images
                      │
                      ▼

         CNN Image Classification
                      │
                      ▼

       Benign / Malignant / Normal
                      │
                      ▼

           Final Diagnostic Insight
```

---

# 📂 Project Structure

```text
lung_cancer_detection/

│
├── app.py
│
├── train_cnn.py
│
├── train_text_model.py
│
├── dataset/
│   ├── lung_cancer.csv
│   └── images/
│       ├── benign/
│       ├── malignant/
│       └── normal/
│
├── models/
│   ├── cnn_model.h5
│   ├── text_model.pkl
│   └── feature_columns.pkl
│
├── requirements.txt
│
└── README.md
```

---

# 🧠 Machine Learning Models

## CNN Model

The image classification model consists of:

- Conv2D Layers
- MaxPooling Layers
- Flatten Layer
- Dense Layers
- Softmax Output

### Image Size

```python
224 × 224 × 3
```

### Optimizer

```python
Adam
```

### Loss Function

```python
Categorical Crossentropy
```

---

## Clinical Prediction Model

### Algorithm

```text
Random Forest Classifier
```

### Input Features

- Age
- Smoking
- Alcohol
- Genetic Risk

### Output

```text
Lung Cancer Risk
Yes / No
```

---

# 📊 Dataset

The project uses:

### Medical CT Scan Images

Classes:

- Benign
- Malignant
- Normal

### Clinical Dataset

Features:

| Feature | Description |
|----------|------------|
| Age | Patient Age |
| Smoking | Smoking History |
| Alcohol | Alcohol Consumption |
| GeneticRisk | Family History |
| Lung_Cancer | Target Variable |

---

# ⚙️ Installation

Clone repository

```bash
git clone https://github.com/Sumeet1021/lung_cancer_detection.git

cd lung_cancer_detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Training CNN Model

```bash
python train_cnn.py
```

Output:

```text
models/cnn_model.h5
```

---

# ▶️ Train Clinical Prediction Model

```bash
python train_text_model.py
```

Output:

```text
models/text_model.pkl
models/feature_columns.pkl
```

---

# 📈 Results

### CNN Model

- Automated CT Scan Classification
- Detects Lung Cancer Patterns
- Supports Multi-Class Prediction

### Clinical Model

- Predicts Lung Cancer Risk
- Uses Patient Health Indicators
- Provides Early Warning Support

---

# 🛠 Tech Stack

### Programming

- Python

### Deep Learning

- TensorFlow
- Keras

### Machine Learning

- Scikit-Learn
- Random Forest

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib

---

# 🎯 Applications

- Medical Diagnosis Support
- Early Lung Cancer Screening
- Healthcare AI Research
- Computer-Aided Diagnosis Systems
- Clinical Decision Support

---

# 💼 Resume Highlights

Developed a Lung Cancer Detection System using Deep Learning and Machine Learning techniques capable of classifying CT scan images and predicting cancer risk from clinical patient data.

Built a CNN-based image classification pipeline using TensorFlow/Keras and a Random Forest model for structured healthcare data analysis.

Implemented automated feature extraction, medical image processing, and predictive analytics to support early cancer detection.

---

# 👨‍💻 Author

### Sumeet Gupta

AI & Data Science Student

Interests:

- Artificial Intelligence
- Healthcare AI
- Deep Learning
- Computer Vision
- Machine Learning
