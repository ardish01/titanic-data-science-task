"""
FastAPI service: logistic regression survival predictions using model_artifacts.joblib.

Prereq: python build_artifacts.py
Run:    uvicorn app:app --reload
         python app.py   (default port 8000, or set PORT=8001 if 8000 is busy)
"""
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_ART = Path(__file__).resolve().parent / "model_artifacts.joblib"
if not _ART.is_file():
    raise FileNotFoundError(
        f"Missing {_ART}. Run: python build_artifacts.py"
    )

artifacts = joblib.load(_ART)
_model = artifacts["model"]
_feature_names = artifacts["feature_names"]
_age_std = artifacts["age_std_scaler"]
_fare_std = artifacts["fare_std_scaler"]
_sex_le = artifacts["sex_label_encoder"]

app = FastAPI(title="Titanic Survival Predictor", version="1.0.0")


class Passenger(BaseModel):
    pclass: int = Field(..., ge=1, le=3)
    age: float = Field(..., ge=0, le=120)
    fare: float = Field(..., ge=0)
    sibsp: int = Field(0, ge=0)
    parch: int = Field(0, ge=0)
    sex: str = Field(..., description="male or female (as in the Titanic dataset)")

    model_config = {"json_schema_extra": {"examples": [{"pclass": 1, "age": 30.0, "fare": 50.0, "sibsp": 0, "parch": 0, "sex": "female"}]}}


def _encode_sex(raw: str) -> int:
    s = raw.strip().lower()
    if s not in ("male", "female"):
        raise HTTPException(status_code=400, detail="sex must be 'male' or 'female'")
    return int(_sex_le.transform([s])[0])


@app.get("/")
def root():
    return {"message": "Titanic Survival Prediction API - POST /predict with passenger JSON"}


@app.post("/predict")
def predict(passenger: Passenger):
    age_s = float(_age_std.transform([[passenger.age]])[0, 0])
    fare_s = float(_fare_std.transform([[passenger.fare]])[0, 0])
    sex_enc = _encode_sex(passenger.sex)
    fam = passenger.sibsp + passenger.parch + 1
    is_alone = int(fam == 1)

    row_df = pd.DataFrame(
        [
            [
                passenger.pclass,
                age_s,
                fare_s,
                passenger.sibsp,
                passenger.parch,
                fam,
                is_alone,
                sex_enc,
            ]
        ],
        columns=_feature_names,
    )
    survived = int(_model.predict(row_df)[0])
    proba = None
    if hasattr(_model, "predict_proba"):
        proba = float(_model.predict_proba(row_df)[0, 1])

    out = {
        "survived": survived,
        "label": "Survived" if survived else "Did not survive",
    }
    if proba is not None:
        out["survival_probability"] = round(proba, 4)
    return out


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
