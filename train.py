import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

GENRES = ["classical", "hiphop", "metal", "jazz", "pop"]
FEATURES = [
    "tempo", "chroma_stft_mean", "rms_mean", "spectral_centroid_mean",
    "spectral_bandwidth_mean", "rolloff_mean", "zero_crossing_rate_mean",
    "harmony_mean", "perceptr_mean", "mfcc1_mean",
]

data_path = Path("features_30_sec.csv")
if not data_path.exists():
    raise FileNotFoundError("features_30_sec.csv not found in project root.")

df = pd.read_csv(data_path)
df = df[df["label"].isin(GENRES)]

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
print("Test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 4))

joblib.dump(model, "model.joblib")

# Save median feature values per genre for sample /predict requests
examples = df.groupby("label")[FEATURES].median().round(6).to_dict("index")
with open("example_requests.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2)