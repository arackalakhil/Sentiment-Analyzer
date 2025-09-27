from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from scipy.stats import mode

# ------------------- Paths -------------------
LOGISTIC_MODEL_PATH = "D:/ml projects vs code/sentence_analyser/logistic_regression_cpu.pkl"
HYBRID_MODEL_PATH   = "D:/ml projects vs code/sentence_analyser/hybrid_cpu_manual.pkl"
VECTORIZER_PATH     = "D:/ml projects vs code/sentence_analyser/tfidf_vectorizer.pkl"

# ------------------- Load models -------------------
logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
hybrid_models  = joblib.load(HYBRID_MODEL_PATH)  # dict of models
vectorizer     = joblib.load(VECTORIZER_PATH)

# ------------------- FastAPI app -------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ------------------- Home -------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ------------------- Combined Prediction -------------------
@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, review_text: str = Form(...)):
    X = vectorizer.transform([review_text])

    # Logistic Prediction
    logistic_pred = logistic_model.predict(X)[0]
    logistic_sentiment = "Good Review" if logistic_pred == 1 else "Bad Review"

    # Hybrid Prediction
    preds = []
    individual_results = {}
    for name, model in hybrid_models.items():
        pred = model.predict(X)[0]
        individual_results[name] = "Good Review" if pred == 1 else "Bad Review"
        preds.append(pred)

    preds = np.array(preds)
    final_pred, _ = mode(preds, axis=0)
    hybrid_sentiment = "Good Review" if final_pred.flatten()[0] == 1 else "Bad Review"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "review": review_text,
            "logistic_sentiment": logistic_sentiment,
            "hybrid_sentiment": hybrid_sentiment,
            "individual_results": individual_results
        }
    )
