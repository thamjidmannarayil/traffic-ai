from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.models import TravelRecord
from app.db.session import SessionLocal
from app.models.request import TravelRecordCreate
from app.models.response import TravelRecordOut
import pandas as pd
import joblib
import os
import logging
from sklearn.ensemble import RandomForestRegressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["Predict Travel Records"])

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "traffic_model.joblib")


# Load trained model once at module import
model = None

def _load_model():
    """Load model from disk into module-level variable."""
    global model
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        logger.info(f"Model exists: {os.path.exists(MODEL_PATH)}")

        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at: {MODEL_PATH}")
            logger.error(f"Project root: {PROJECT_ROOT}")
            logger.error(f"Directory contents: {os.listdir(PROJECT_ROOT)}")
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        model = joblib.load(MODEL_PATH)
        logger.info("✓ Model loaded successfully!")
        if hasattr(model, 'feature_names_in_'):
            logger.info(f"Model features ({len(model.feature_names_in_)}): {list(model.feature_names_in_)}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        model = None


_load_model()


def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def preprocess_input(record_dict: dict):
    """
    Preprocess input data to match the format expected by the ML model.

    Args:
        record_dict: Dictionary containing the travel record data

    Returns:
        DataFrame with preprocessed features ready for prediction
    """
    if model is None:
        raise ValueError("Model is not loaded")

    # Create DataFrame from input
    df = pd.DataFrame([record_dict])

    # Encode time_of_day as minutes since midnight with strict validation
    try:
        parsed_times = pd.to_datetime(df["time_of_day"], format="%H:%M")
    except Exception as exc:
        raise ValueError("time_of_day must be in 'HH:MM' 24-hour format") from exc

    df["time_minutes"] = parsed_times.dt.hour * 60 + parsed_times.dt.minute
    df = df.drop(columns=["time_of_day"])

    # Convert festival boolean to int (don't one-hot encode it)
    if "festival" in df.columns:
        df["festival"] = df["festival"].astype(int)

    # One-hot encode categorical columns (festival is boolean, not categorical)
    cat_cols = ["weather", "road_type", "day_of_week"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Ensure all model features exist and are in the correct order
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Loaded model is missing feature_names_in_. Retrain the model with scikit-learn >= 1.0.")

    model_features = list(model.feature_names_in_)

    # Reindex will add any missing columns with 0 and order them to match the model
    df = df.reindex(columns=model_features, fill_value=0)

    return df


def _boolify(value):
    if isinstance(value, bool):
        return int(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    text_val = str(value).strip().lower()
    if text_val in {"1", "true", "yes", "y", "t"}:
        return 1
    if text_val in {"0", "false", "no", "n", "f"}:
        return 0
    # Fallback: truthiness
    return int(bool(value))


def preprocess_training_dataframe(df: pd.DataFrame):
    """
    Prepare full training dataframe into feature matrix X and target y.
    Raises ValueError if required columns are missing or invalid.
    """
    # Normalize column names to plain strings to avoid quoted_name type mix
    df = df.copy()
    df.columns = [str(col) for col in df.columns]

    required_cols = {
        "distance_km",
        "congestion_index",
        "time_of_day",
        "weather",
        "road_type",
        "day_of_week",
        "festival",
        "actual_travel_time_min",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Training data missing required columns: {', '.join(sorted(missing))}")

    # Validate and transform time_of_day
    parsed_times = pd.to_datetime(df["time_of_day"], format="%H:%M", errors="coerce")
    if parsed_times.isna().any():
        bad_rows = df.loc[parsed_times.isna(), "time_of_day"].unique()
        raise ValueError(f"Invalid time_of_day format encountered: {bad_rows}")
    df = df.copy()
    df["time_minutes"] = parsed_times.dt.hour * 60 + parsed_times.dt.minute
    df = df.drop(columns=["time_of_day"])

    # Festival to int
    df["festival"] = df["festival"].apply(_boolify).astype(int)

    # Separate target
    y = df["actual_travel_time_min"].astype(float)
    feature_df = df.drop(columns=["actual_travel_time_min", "id", "date", "source", "destination"], errors="ignore")

    cat_cols = ["weather", "road_type", "day_of_week"]
    X = pd.get_dummies(feature_df, columns=cat_cols, drop_first=False)
    X.columns = [str(col) for col in X.columns]

    return X, y


@router.post("/train", response_model=dict)
def train_model_endpoint(db: Session = Depends(get_db)):
    """
    Train a new model on all stored travel records and replace traffic_model.joblib.
    """
    # 1) Load data
    try:
        engine = db.get_bind()
        df = pd.read_sql_table(TravelRecord.__tablename__, con=engine)
        # Normalize column names early to avoid sqlalchemy.quoted_name types
        df.columns = [str(col) for col in df.columns]
    except Exception as exc:
        logger.error(f"Failed to fetch training data: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to load training data: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="No data available to train the model.")

    # 2) Preprocess
    try:
        X, y = preprocess_training_dataframe(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 3) Train
    try:
        new_model = RandomForestRegressor(n_estimators=200, random_state=42)
        new_model.fit(X, y)
    except Exception as exc:
        logger.error(f"Model training failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {exc}")

    # 4) Persist
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(new_model, MODEL_PATH)
    except Exception as exc:
        logger.error(f"Failed to save trained model: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to save trained model: {exc}")

    # 5) Hot reload in memory
    global model
    model = new_model
    logger.info("✓ Model retrained and saved.")

    return {
        "message": "Model retrained and saved.",
        "records_used": int(len(df)),
        "feature_count": int(len(new_model.feature_names_in_)),
    }


@router.post("/", response_model=TravelRecordOut)
def predict_data(record: TravelRecordCreate, db: Session = Depends(get_db)):
    """
    Predict travel time for a given route and save to database.

    Args:
        record: Travel record data (without predicted travel time)
        db: Database session

    Returns:
        Complete travel record with predicted travel time
    """
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Model failed to load. Check server logs for details."
            )

        logger.info(f"Received prediction request for route: {record.source} -> {record.destination}")

        # Convert to dict and remove actual_travel_time_min if it exists
        # (since we're predicting it, user shouldn't provide it)
        record_dict = record.model_dump()
        if "actual_travel_time_min" in record_dict:
            record_dict.pop("actual_travel_time_min")

        # Preprocess input for prediction
        try:
            X = preprocess_input(record_dict)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        logger.info(f"Preprocessed features shape: {X.shape}")

        # Make prediction
        predicted_time = float(model.predict(X)[0])
        logger.info(f"Predicted travel time: {predicted_time:.2f} minutes")

        # Save to DB with predicted travel time
        db_record = TravelRecord(
            **record_dict,
            actual_travel_time_min=predicted_time
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        logger.info(f"✓ Record saved with ID: {db_record.id}")
        return db_record

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
