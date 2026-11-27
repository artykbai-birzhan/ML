# Music Genre Classifier: FastAPI + Docker + Streamlit + MLflow

A Random Forest model predicts a music genre (`classical`, `hiphop`, `metal`, `jazz`, `pop`)
from 10 audio features. The model is served with FastAPI (Docker), used through a Streamlit
frontend, and tracked and registered with MLflow.

**Dataset:** [GTZAN Dataset - Music Genre Classification (Kaggle)](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
Download `features_30_sec.csv` and put it in the project root.

## Project structure

```
ml-fastapi-docker/
├── main.py                # FastAPI app: GET /, POST /predict
├── app.py                 # Streamlit frontend (calls the API)
├── train.py               # trains the model, logs to MLflow, registers it, saves model.joblib
├── features_30_sec.csv    # dataset (download from Kaggle)
├── model.joblib           # saved model (created by train.py)
├── example_requests.json  # example inputs per genre (created by train.py)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup and run

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py                   # trains, logs to MLflow, registers the model
```

Run each of these in a separate terminal:

```bash
uvicorn main:app --reload                          # API:       http://127.0.0.1:8000/docs
streamlit run app.py                               # Frontend:  http://localhost:8501
mlflow ui --backend-store-uri sqlite:///mlflow.db  # MLflow UI: http://127.0.0.1:5000
```

## API

- `GET /` returns `{"message": "ML API is running"}`
- `POST /predict` takes 10 audio features and returns the genre and class probabilities.
  Ready-to-use inputs for each genre are in `example_requests.json`.

## Streamlit frontend

Pick an example genre (or edit the 10 values by hand) and press **Predict**. The app sends the values
to the API and shows the predicted genre with a probability chart. By default it uses
`http://127.0.0.1:8000`; set the `API_URL` environment variable to use another address.

## MLflow

`train.py` creates the experiment `music-genre-classification` and logs, per run:

- **parameters:** `n_estimators`, `random_state`
- **metrics:** `accuracy`, `f1_macro`
- **artifacts:** `model.joblib` and the MLflow model (`model/`)

The model is registered in the Model Registry as `music-genre-classifier`.
Each `python train.py` run creates a new version (1, 2, 3, ...).
Open the MLflow UI to see the experiment under **Experiments** and the versions under **Models**.

## Docker

```bash
docker build -t ml-fastapi-app .
docker run -p 8000:8000 ml-fastapi-app
```

The container serves the API at http://localhost:8000/docs, and `streamlit run app.py` works with it as is.
The image re-trains the model during the build, so `features_30_sec.csv` must be in the project folder.
