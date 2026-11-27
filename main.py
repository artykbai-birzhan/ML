from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="Music Genre API")

MODEL_PATH = Path("model.joblib")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class AudioFeatures(BaseModel):
    tempo: float
    chroma_stft_mean: float
    rms_mean: float
    spectral_centroid_mean: float
    spectral_bandwidth_mean: float
    rolloff_mean: float
    zero_crossing_rate_mean: float
    harmony_mean: float
    perceptr_mean: float
    mfcc1_mean: float

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "tempo": 123.05,
                    "chroma_stft_mean": 0.335,
                    "rms_mean": 0.13,
                    "spectral_centroid_mean": 1773.0,
                    "spectral_bandwidth_mean": 1972.0,
                    "rolloff_mean": 3806.0,
                    "zero_crossing_rate_mean": 0.084,
                    "harmony_mean": 0.0,
                    "perceptr_mean": 0.0,
                    "mfcc1_mean": -113.6,
                }
            ]
        }
    )


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: AudioFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="model.joblib not found. Run train.py first.")

    x = pd.DataFrame([features.model_dump()])[model.feature_names_in_]
    probs = model.predict_proba(x)[0]
    pred_idx = int(probs.argmax())
    return {
        "prediction": str(model.classes_[pred_idx]),
        "probabilities": {c: round(float(p), 4) for c, p in zip(model.classes_, probs)},
    }
