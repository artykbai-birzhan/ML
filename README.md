# Music Genre Classifier: FastAPI + Docker

A Random Forest model predicts a music genre (`classical`, `hiphop`, `metal`, `jazz`, `pop`)
from 10 audio features and is served as a REST API with FastAPI and Docker.

**Dataset:** [GTZAN Dataset - Music Genre Classification (Kaggle)](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
Download `features_30_sec.csv` and put it in the project root.

## Project structure

```
ml-fastapi-docker/
├── main.py                # FastAPI app: GET /, POST /predict
├── train.py               # trains the model, saves model.joblib and example_requests.json
├── features_30_sec.csv    # dataset (download from Kaggle)
├── model.joblib           # saved model (created by train.py)
├── example_requests.json  # ready-to-use /predict inputs for each genre
├── requirements.txt
├── Dockerfile
└── README.md
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```

- API: http://127.0.0.1:8000/
- Swagger docs: http://127.0.0.1:8000/docs

## Endpoints

**GET /**
```json
{"message": "ML API is running"}
```

**POST /predict**: send the 10 features (take a ready example from `example_requests.json`):
```json
{
  "tempo": 123.05, "chroma_stft_mean": 0.335, "rms_mean": 0.13,
  "spectral_centroid_mean": 1773.0, "spectral_bandwidth_mean": 1972.0,
  "rolloff_mean": 3806.0, "zero_crossing_rate_mean": 0.084,
  "harmony_mean": 0.0, "perceptr_mean": 0.0, "mfcc1_mean": -113.6
}
```
Response:
```json
{"prediction": "jazz", "probabilities": {"classical": 0.1, "hiphop": 0.1, "jazz": 0.5, "metal": 0.1, "pop": 0.2}}
```

## Run with Docker

```bash
docker build -t ml-fastapi-app .
docker run -p 8000:8000 ml-fastapi-app
```

Open http://localhost:8000/docs and test `/` and `/predict` again.
The image re-trains the model during the build, so `features_30_sec.csv` must be in the project folder.
