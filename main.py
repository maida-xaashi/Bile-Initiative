from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import joblib
import pandas as pd
from helpers.api_helper import StudentInputData,get_student_prediction_result


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = "student-at-risk-model.joblib"
model = joblib.load(BASE_DIR / MODEL_FILE)

app = FastAPI(
    title="Student At Risk System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get('/api')
def api_home():
    return {"message": "Welcome to the Student At Risk System API!"}


@app.get('/model-info')
def model_info():
    final_model = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    return {
        "model_file": MODEL_FILE,
        "model_type": final_model.__class__.__name__,
        "features": list(getattr(model, "feature_names_in_", [])),
    }


@app.post("/predict")
def predict_student_at_risk(student_data: StudentInputData):
    data_frame = pd.DataFrame([student_data.model_dump()])
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None:
        data_frame = data_frame[list(expected_features)]
    prediction = model.predict(data_frame)[0]
    prob = model.predict_proba(data_frame)[0]
    result = get_student_prediction_result(prediction, prob)
    final_model = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    result["model"] = {
        "file": MODEL_FILE,
        "type": final_model.__class__.__name__,
        "raw_prediction": int(prediction),
    }
    return result
