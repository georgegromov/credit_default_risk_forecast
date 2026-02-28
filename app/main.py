from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pickle
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

app = FastAPI(title="Credit Default Prediction API")

# Загрузка модели и вспомогательных файлов
MODEL_PATH = os.getenv("MODEL_PATH", "/model/credit_default_model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "/model/scaler.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "/model/feature_names.json")

model = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)


class ClientData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.get("/", response_class=HTMLResponse)
def root():
    with open("/app/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
def predict(data: ClientData):
    try:
        input_array = np.array([[getattr(data, f) for f in feature_names]])
        scaled = scaler.transform(input_array)
        prob = float(model.predict(scaled, verbose=0)[0][0])
        result = "Дефолт" if prob >= 0.5 else "Нет дефолта"
        risk = "Высокий" if prob >= 0.5 else ("Средний" if prob >= 0.3 else "Низкий")
        return {
            "prediction": result,
            "probability": round(prob, 4),
            "risk_level": risk
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}