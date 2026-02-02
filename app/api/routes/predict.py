from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.models import TravelRecord
from app.db.session import SessionLocal
from app.models.request import TravelRecordCreate
from app.models.response import TravelRecordOut
import pandas as pd
import joblib

router = APIRouter(prefix="/predictions", tags=["Predict Travel Records"])

# Load trained model once
import os
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "traffic_model.joblib")
model = joblib.load(model_path)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper function to preprocess input like training
def preprocess_input(record):
    df = pd.DataFrame([record])

    # Encode time_of_day as minutes since midnight
    df["hour"] = df["time_of_day"].str.split(":").apply(lambda x: int(x[0]))
    df["minute"] = df["time_of_day"].str.split(":").apply(lambda x: int(x[1]))
    df["time_minutes"] = df["hour"] * 60 + df["minute"]
    df = df.drop(columns=["time_of_day", "hour", "minute"])

    # One-hot encode categorical columns (festival is boolean, not categorical)
    cat_cols = ["weather", "road_type", "day_of_week"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Convert festival boolean to int if needed
    if "festival" in df.columns:
        df["festival"] = df["festival"].astype(int)

    # Ensure all model features exist
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]  # reorder to match model
    return df


# Create & predict
@router.post("/", response_model=TravelRecordOut)
def predict_data(record: TravelRecordCreate, db: Session = Depends(get_db)):
    # Preprocess input for prediction
    X = preprocess_input(record.model_dump())
    predicted_time = float(model.predict(X)[0])

    # Save to DB including predicted travel time
    db_record = TravelRecord(**record.model_dump(), actual_travel_time_min=predicted_time)
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record
