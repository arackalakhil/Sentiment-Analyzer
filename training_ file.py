import os
import joblib
import cudf
import pandas as pd
from cuml.feature_extraction.text import TfidfVectorizer as cuTfidf
import cupy as cp

# ---------------- FILE PATHS ----------------
TRAIN_CSV = "/content/drive/MyDrive/sentimental_data/train.csv"
TEST_CSV  = "/content/drive/MyDrive/sentimental_data/test.csv"

VECTORIZER_FILE = "/content/drive/MyDrive/sentimental_data/tfidf_vectorizer_gpu.pkl"
X_TRAIN_FILE    = "/content/drive/MyDrive/sentimental_data/X_train_gpu.pkl"
X_TEST_FILE     = "/content/drive/MyDrive/sentimental_data/X_test_gpu.pkl"

# ---------------- GPU CHECK ----------------
def check_gpu():
    if cp.cuda.runtime.getDeviceCount() > 0:
        print("✅ GPU detected:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    else:
        print("⚠️ No GPU found! Check Colab settings (Runtime → Change runtime type → GPU)")

check_gpu()

# ---------------- LOAD DATA ----------------
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print("Train size:", train_df.shape, "Test size:", test_df.shape)
print(train_df.head())

# ✅ GPU detected: Tesla T4
# Train size: (3600000, 2) Test size: (400000, 2)
import re
import cudf

# Convert pandas to cudf for GPU acceleration
train_gdf = cudf.from_pandas(train_df)
test_gdf  = cudf.from_pandas(test_df)

# Example text column: 'text'
def clean_text_gpu(text_series):
    # Lowercase
    text_series = text_series.str.lower()
    # Remove special characters and numbers
    text_series = text_series.str.replace(r"[^a-z\s]", "", regex=True)
    # Remove extra spaces
    text_series = text_series.str.replace(r"\s+", " ", regex=True)
    text_series = text_series.str.strip()
    return text_series

train_gdf['clean_text'] = clean_text_gpu(train_gdf['text'])
test_gdf['clean_text']  = clean_text_gpu(test_gdf['text'])

print(train_gdf[['text', 'clean_text']].head())
# Cleaned text examples
# 0  stuning even for the non gamer this sound trac...   
# 1  the best soundtrack ever to anything i m readi...   
# 2  amazing this soundtrack is my favorite music o...   
# 3  excellent soundtrack i truly like this soundtr...   
# 4  remember pull your jaw off the floor after hea...   

#                                           clean_text  
# 0  stuning even for the non gamer this sound trac...  
# 1  the best soundtrack ever to anything i m readi...  
# 2  amazing this soundtrack is my favorite music o...  
# 3  excellent soundtrack i truly like this soundtr...  
# 4  remember pull your jaw off the floor after hea...
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import cupyx.scipy.sparse as cps

# ---------------- TF-IDF VECTORIZE IN BATCHES ----------------
vectorizer = TfidfVectorizer(max_features=20000, min_df=5, max_df=0.8)
batch_size = 500000
X_train_list = []

# Fill missing values in clean_text with empty string
train_gdf['clean_text'] = train_gdf['clean_text'].fillna("")
test_gdf['clean_text']  = test_gdf['clean_text'].fillna("")

# Process training data in batches
for i in range(0, len(train_gdf), batch_size):
    batch_text = train_gdf['clean_text'].iloc[i:i+batch_size].to_pandas()

    # Convert everything to string (just in case)
    batch_text = batch_text.astype(str)

    if i == 0:
        X_batch_cpu = vectorizer.fit_transform(batch_text)
    else:
        X_batch_cpu = vectorizer.transform(batch_text)

    # Convert CPU sparse matrix to GPU sparse matrix
    X_batch_gpu = cps.csr_matrix(X_batch_cpu)
    X_train_list.append(X_batch_gpu)

# Stack all batches vertically on GPU
X_train_gpu = cps.vstack(X_train_list)

# ---------------- PROCESS TEST DATA ----------------
X_test_cpu = vectorizer.transform(test_gdf['clean_text'].to_pandas().astype(str))
X_test_gpu = cps.csr_matrix(X_test_cpu)

# ---------------- SAVE VECTORIZER AND MATRICES ----------------
joblib.dump(vectorizer, "/content/drive/MyDrive/sentimental_data/tfidf_vectorizer_gpu.pkl")
joblib.dump(X_train_gpu, "/content/drive/MyDrive/sentimental_data/X_train_gpu.pkl")
joblib.dump(X_test_gpu, "/content/drive/MyDrive/sentimental_data/X_test_gpu.pkl")

print("✅ TF-IDF batch-wise GPU conversion done.")
print("Train shape:", X_train_gpu.shape, "Test shape:", X_test_gpu.shape)
# ✅ TF-IDF batch-wise GPU conversion done.
# Train shape: (3600000, 20000) Test shape: (400000, 20000)
from cuml.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cupy as cp

# ------------------- Prepare Validation -------------------
val_idx = int(0.9 * X_train_gpu.shape[0])
X_train_split = X_train_gpu[:val_idx]
y_train_split = y_train[:val_idx]
X_val_split   = X_train_gpu[val_idx:]
y_val_split   = y_train[val_idx:]

print("Training shape:", X_train_split.shape)
print("Validation shape:", X_val_split.shape)

# ------------------- Logistic Regression -------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_split, y_train_split)

# ------------------- Validation Predictions -------------------
y_val_pred = lr.predict(X_val_split)

# Convert GPU arrays to NumPy for sklearn metrics
y_val_pred_np = cp.asnumpy(y_val_pred)
y_val_np      = cp.asnumpy(y_val_split)

# Compute metrics on CPU
acc = accuracy_score(y_val_np, y_val_pred_np)
prec = precision_score(y_val_np, y_val_pred_np)
rec = recall_score(y_val_np, y_val_pred_np)
f1 = f1_score(y_val_np, y_val_pred_np)

print("\n✅ Logistic Regression Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")

# ------------------- Test Predictions -------------------
y_test_pred = lr.predict(X_test_gpu)
y_test_pred_np = cp.asnumpy(y_test_pred)
print("\n✅ Test predictions done. Shape:", y_test_pred_np.shape)
# Training shape: (3240000, 20000)
# Validation shape: (360000, 20000)

# ✅ Logistic Regression Metrics:
# Accuracy: 0.9109
# Precision: 0.9090
# Recall: 0.9107
# F1-Score: 0.9098

# ✅ Test predictions done. Shape: (400000,)
import joblib

# Path to save on Google Drive
MODEL_FILE = "/content/drive/MyDrive/sentimental_data/logistic_regression_gpu.pkl"

# Save the trained Logistic Regression model
joblib.dump(lr, MODEL_FILE)

print(f"✅ Model saved to: {MODEL_FILE}")
# ✅ Model saved to: /content/drive/MyDrive/sentimental_data/logistic_regression_gpu.pkl
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# GPU libraries
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.svm import SVC as cuSVC
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.ensemble import VotingClassifier  # wrapper works with cuML too
import cupy as cp

# ------------------- Reload Data -------------------
VECTORIZER_FILE = "/content/drive/MyDrive/sentimental_data/tfidf_vectorizer.pkl"
vectorizer = joblib.load(VECTORIZER_FILE)

TRAIN_CSV = "/content/drive/MyDrive/sentimental_data/train.csv"
TEST_CSV  = "/content/drive/MyDrive/sentimental_data/test.csv"

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# Replace NaN with blank
train_df['text'] = train_df['text'].fillna("")
test_df['text']  = test_df['text'].fillna("")

# Transform using vectorizer
X_train_cpu = vectorizer.transform(train_df['text'])
y_train_cpu = train_df['label'].values
X_test_cpu  = vectorizer.transform(test_df['text'])

print("CPU Train shape:", X_train_cpu.shape, "Test shape:", X_test_cpu.shape)

# Convert to GPU arrays
import cupyx
X_train_gpu = cupyx.scipy.sparse.csr_matrix(X_train_cpu)
X_test_gpu  = cupyx.scipy.sparse.csr_matrix(X_test_cpu)
y_train_gpu = cp.asarray(y_train_cpu)

# ------------------- GPU Models -------------------
lr_gpu  = cuLogisticRegression(max_iter=1000)
rf_gpu  = cuRF(n_estimators=100, max_depth=20, random_state=42)
svm_gpu = cuSVC(kernel='linear', C=1.0, probability=True)

hybrid_gpu = VotingClassifier(
    estimators=[('lr', lr_gpu), ('rf', rf_gpu), ('svm', svm_gpu)],
    voting='hard'
)

# ------------------- Train on GPU -------------------
hybrid_gpu.fit(X_train_gpu, y_train_gpu)

# ------------------- Validation -------------------
val_idx = int(0.9 * X_train_gpu.shape[0])
X_val_gpu = X_train_gpu[val_idx:]
y_val_gpu = y_train_gpu[val_idx:]

y_val_pred_gpu = hybrid_gpu.predict(X_val_gpu).get()

acc = accuracy_score(y_val_gpu.get(), y_val_pred_gpu)
prec = precision_score(y_val_gpu.get(), y_val_pred_gpu)
rec = recall_score(y_val_gpu.get(), y_val_pred_gpu)
f1 = f1_score(y_val_gpu.get(), y_val_pred_gpu)

print("\n✅ GPU Hybrid Ensemble Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")

# ------------------- Save GPU model -------------------
HYBRID_GPU_FILE = "/content/drive/MyDrive/sentimental_data/hybrid_model_gpu.pkl"
joblib.dump(hybrid_gpu, HYBRID_GPU_FILE)
print(f"✅ GPU Hybrid model saved to: {HYBRID_GPU_FILE}")

# ------------------- Also train CPU version -------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

lr_cpu  = LogisticRegression(max_iter=1000)
rf_cpu  = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
svm_cpu = SVC(kernel='linear', C=1.0, probability=True)

hybrid_cpu = VotingClassifier(
    estimators=[('lr', lr_cpu), ('rf', rf_cpu), ('svm', svm_cpu)],
    voting='hard'
)

hybrid_cpu.fit(X_train_cpu, y_train_cpu)

# Save CPU model
HYBRID_CPU_FILE = "/content/drive/MyDrive/sentimental_data/hybrid_model_cpu.pkl"
joblib.dump(hybrid_cpu, HYBRID_CPU_FILE)
print(f"✅ CPU Hybrid model saved to: {HYBRID_CPU_FILE}")
# ✅ TF-IDF vectorizer saved to: /content/drive/MyDrive/sentimental_data/tfidf_vectorizer.pkl
# ✅ GPU Logistic Regression model saved to: /content/drive/MyDrive/sentimental_data/logistic_regression_gpu.pkl
# ✅ CPU Logistic Regression model saved to: /content/drive/MyDrive/sentimental_data/logistic_regression_cpu.pkl

# ✅ CPU Logistic Regression Metrics:
# Accuracy: 0.9115
# Precision: 0.9102
# Recall: 0.9105
# F1-Score: 0.9104

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from scipy import sparse

# ------------------- Paths -------------------
VECTORIZER_FILE = "/content/drive/MyDrive/sentimental_data/tfidf_vectorizer.pkl"
TRAIN_CSV       = "/content/drive/MyDrive/sentimental_data/train.csv"
TEST_CSV        = "/content/drive/MyDrive/sentimental_data/test.csv"
HYBRID_CPU_FILE = "/content/drive/MyDrive/sentimental_data/hybrid_cpu_manual.pkl"

# ------------------- Load Vectorizer & Data -------------------
vectorizer = joblib.load(VECTORIZER_FILE)

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

train_df['text'] = train_df['text'].fillna("")
test_df['text']  = test_df['text'].fillna("")

# ------------------- Transform Data -------------------
X_train = vectorizer.transform(train_df['text'])
y_train = train_df['label'].values
X_test  = vectorizer.transform(test_df['text'])

print("✅ CPU Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ------------------- Initialize CPU Models -------------------
lr_cpu  = LogisticRegression(max_iter=1000)
svm_cpu = LinearSVC(max_iter=1000)
nb_cpu  = MultinomialNB()

# ------------------- Train CPU Models -------------------
print("⏳ Training LogisticRegression...")
lr_cpu.fit(X_train, y_train)

print("⏳ Training LinearSVC...")
svm_cpu.fit(X_train, y_train)

print("⏳ Training MultinomialNB...")
nb_cpu.fit(X_train, y_train)

# ------------------- Manual Hard Voting -------------------
def hard_vote(preds_list):
    """
    preds_list: list of np.ndarray predictions from each model
    returns majority vote prediction
    """
    stacked = np.vstack(preds_list)
    votes = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, stacked)
    return votes

# ------------------- Validation -------------------
val_idx = int(0.9 * X_train.shape[0])
X_val = X_train[val_idx:]
y_val = y_train[val_idx:]

preds_lr  = lr_cpu.predict(X_val)
preds_svm = svm_cpu.predict(X_val)
preds_nb  = nb_cpu.predict(X_val)

y_val_pred = hard_vote([preds_lr, preds_svm, preds_nb])

# ------------------- Compute Metrics -------------------
acc  = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec  = recall_score(y_val, y_val_pred)
f1   = f1_score(y_val, y_val_pred)

print("\n📊 CPU Hybrid Metrics (LogReg + LinearSVC + MultinomialNB):")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")

# ------------------- Save CPU Models -------------------
joblib.dump({'lr': lr_cpu, 'svm': svm_cpu, 'nb': nb_cpu}, HYBRID_CPU_FILE)
print(f"✅ CPU Hybrid models saved to: {HYBRID_CPU_FILE}")
# ✅ CPU Train shape: (3600000, 20000) Test shape: (400000, 20000)
# ⏳ Training LogisticRegression...
# ⏳ Training LinearSVC...
# ⏳ Training MultinomialNB...

# 📊 CPU Hybrid Metrics (LogReg + LinearSVC + MultinomialNB):
# Accuracy : 0.9120
# Precision: 0.9108
# Recall   : 0.9110
# F1-Score : 0.9109
# ✅ CPU Hybrid models saved to: /content/drive/MyDrive/sentimental_data/hybrid_cpu_manual.pkl
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import mode

# 🔹 Paths
MODEL_PATH      = "/content/drive/MyDrive/sentimental_data/hybrid_cpu_manual.pkl"
VECTORIZER_PATH = "/content/drive/MyDrive/sentimental_data/tfidf_vectorizer.pkl"
TEST_PATH       = "/content/drive/MyDrive/sentimental_data/test.csv"

# 🔹 Load test data
print("📂 Loading test data...")
test_df = pd.read_csv(TEST_PATH)

# Assuming test.csv has columns: ["label", "text"]
X_test = test_df["text"].fillna("")
y_test = test_df["label"]

# 🔹 Load vectorizer and models dict
print("📦 Loading vectorizer + models...")
vectorizer = joblib.load(VECTORIZER_PATH)
models     = joblib.load(MODEL_PATH)   # This should be a dict: {"log_reg": ..., "svc": ..., "nb": ...}

# 🔹 Vectorize test data
X_test_vec = vectorizer.transform(X_test)

# 🔹 Run predictions from each model
preds = []
for name, model in models.items():
    print(f"🤖 Predicting with {name}...")
    preds.append(model.predict(X_test_vec))

# Convert list of predictions into array [n_models, n_samples]
preds = np.array(preds)

# 🔹 Hard voting (majority rule)
y_pred, _ = mode(preds, axis=0)
y_pred = y_pred.flatten()

# 🔹 Evaluate
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
rec  = recall_score(y_test, y_pred, average="weighted")
f1   = f1_score(y_test, y_pred, average="weighted")

print("\n📊 Final Test Set Metrics:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")

# 🔹 Save predictions (optional)
out_path = "/content/drive/MyDrive/sentimental_data/test_predictions.csv"
pd.DataFrame({
    "text": X_test,
    "true_label": y_test,
    "pred_label": y_pred
}).to_csv(out_path, index=False)

print(f"\n✅ Predictions saved to: {out_path}")
# 📂 Loading test data...
# 📦 Loading vectorizer + models...
# 🤖 Predicting with lr...
# 🤖 Predicting with svm...
# 🤖 Predicting with nb...

# 📊 Final Test Set Metrics:
# Accuracy : 0.9089
# Precision: 0.9089
# Recall   : 0.9089
# F1-Score : 0.9089

# ✅ Predictions saved to: /content/drive/MyDrive/sentimental_data/test_predictions.csv