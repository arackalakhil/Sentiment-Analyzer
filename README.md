# Sentiment Analyzer

A FastAPI web application that predicts sentiment of text reviews using **Logistic Regression** and a **Hybrid Ensemble Model** (Logistic Regression + LinearSVC + MultinomialNB). It classifies reviews as **Good Review** or **Bad Review**.

---

## Project Structure

sentence_analyser/
│
├── main.py # FastAPI backend
├── templates/
│ └── index.html # Web interface
├── logistic_regression_cpu.pkl # Trained Logistic Regression model
├── hybrid_cpu_manual.pkl # Trained Hybrid ensemble model
├── tfidf_vectorizer.pkl # Trained TF-IDF vectorizer
└── README.md # Project documentation

---

## Features

- Predict sentiment using:
  - Logistic Regression (CPU)
  - Hybrid Ensemble (Logistic Regression + LinearSVC + MultinomialNB)
- Shows individual predictions from the Hybrid ensemble.
- Simple web interface using FastAPI + Jinja2.
- Optionally saves predictions to CSV.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Sentiment-Analyzer.git
cd Sentiment-Analyzer

2.Create and activate a virtual environment:v
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt


Example requirements.txt:

fastapi
uvicorn
jinja2
scikit-learn
scipy
numpy
pandas
joblib

Running the Application
uvicorn main:app --reload


Open your browser at http://127.0.0.1:8000

Enter a review and see predictions from both Logistic Regression and Hybrid Ensemble models.

How It Works

Vectorization: Input text is converted to TF-IDF features using a pre-trained vectorizer.

Prediction:

Logistic Regression predicts sentiment.

Hybrid ensemble predicts via majority voting over Logistic Regression, LinearSVC, and MultinomialNB.

Output: Displays Logistic Regression result, Hybrid ensemble result, and individual model votes.

Example Usage

Input review:

The service was excellent and the food was delicious!


Output:

Logistic Regression: Good Review

Hybrid Ensemble: Good Review

Individual model votes:

lr: Good Review

svm: Good Review

nb: Good Review

Notes

Ensure model and vectorizer paths in main.py are correct.

CPU-only setup; GPU version requires cuML/RAPIDS.

Binary sentiment classification: 1 = Good Review, 0 = Bad Review.

References

FastAPI Documentation

Scikit-learn VotingClassifier

TF-IDF Vectorization