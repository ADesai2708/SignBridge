#  SignBridge

SignBridge is a personal project that I'm building to explore Computer Vision, Machine Learning, and Natural Language Processing by solving a real-world accessibility problem.

The goal of this project is to create a bidirectional sign language translator that can:

- Convert sign language into text in real time.
- Convert text into sign language using sign videos.

I'm building this project step by step instead of using an end-to-end deep learning model because I want to understand every part of the pipeline, from hand detection to model training and deployment.

---

## Current Progress

### ✅ Completed

- Webcam integration using OpenCV
- Hand detection using MediaPipe
- Extraction of 21 hand landmarks (63 features)
- Landmark normalization
- Dataset preprocessing pipeline
- Landmark dataset generation (CSV)
- Static sign classifier using Scikit-learn MLP
- Model training and evaluation
- Live webcam prediction
- Top-3 prediction support

Current model accuracy (A, B and C classes): **99.75%**

---

## Project Workflow

```
Webcam/Image
      │
      ▼
MediaPipe Hand Detection
      │
      ▼
Landmark Extraction
      │
      ▼
Feature Normalization
      │
      ▼
MLP Classifier
      │
      ▼
Predicted Letter
```

---

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Joblib

Planned:

- PyTorch
- Streamlit
- Anthropic Claude API
- Hugging Face Spaces

---

## Project Structure

```
SignBridge
│
├── app/
├── data/
├── models/
├── src/
│   ├── extraction/
│   ├── inference/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   └── utils/
│
├── requirements.txt
└── README.md
```

---

## Dataset

I'm using the **ASL Alphabet Dataset** from Kaggle for the static sign recognition model.

Current training is done on the letters:

- A
- B
- C

The preprocessing pipeline is already designed to support all classes.

---

## What I Learned While Building This

This project helped me understand:

- How MediaPipe detects and tracks hands
- How landmark-based feature extraction works
- Data preprocessing for machine learning
- Feature engineering
- Training and evaluating an MLP classifier
- Saving and loading trained models
- Running real-time inference with a webcam

---

## Next Steps

I'm currently working on:

- Confidence thresholding
- Duplicate prediction filtering
- Letter buffering
- Word formation
- Sentence generation

After that I'll add:

- Dynamic sign recognition using LSTM
- Text to Sign conversion
- Streamlit interface
- ISL support
- Hugging Face deployment

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/<your-username>/SignBridge.git
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/training/train_static.py
```

Run live prediction:

```bash
python src/inference/predict_live.py
```

---

## Why I Started This Project

Most sign language projects either recognize only alphabets or rely on large deep learning models without explaining the process.

I wanted to build the complete system myself and understand each component individually. The long-term goal is to support both static and dynamic signs and create a practical tool that can help improve communication accessibility.

---

## Author

**Anjali Desai**

B.Tech CSE (AI & ML)

This project is still under active development, and I'll continue updating it as I add new features.
