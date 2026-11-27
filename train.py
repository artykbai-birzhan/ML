import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

GENRES = ["classical", "hiphop", "metal", "jazz", "pop"]
FEATURES = [
    "tempo", "chroma_stft_mean", "rms_mean", "spectral_centroid_mean",
    "spectral_bandwidth_mean", "rolloff_mean", "zero_crossing_rate_mean",
    "harmony_mean", "perceptr_mean", "mfcc1_mean",
]
PARAMS = {"n_estimators": 300, "random_state": 42}

df = pd.read_csv("features_30_sec.csv")
df = df[df["label"].isin(GENRES)]

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("music-genre-classification")

with mlflow.start_run():
    model = RandomForestClassifier(**PARAMS).fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print("Test accuracy:", round(accuracy, 4))

    mlflow.log_params(PARAMS)
    mlflow.log_metrics({"accuracy": accuracy, "f1_macro": f1_score(y_test, pred, average="macro")})

    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib")
    mlflow.sklearn.log_model(
        model, name="model", registered_model_name="music-genre-classifier",
        serialization_format="cloudpickle",
    )

# One ready-to-use /predict request per genre (median feature values)
examples = df.groupby("label")[FEATURES].median().round(6).to_dict("index")
json.dump(examples, open("example_requests.json", "w"), indent=2)
