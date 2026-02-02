from datetime import date
import os

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.api.routes import predict
from app.db import models


def setup_in_memory_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    models.Base.metadata.create_all(bind=engine)
    return engine, TestingSessionLocal


def test_train_endpoint_trains_and_saves_model(tmp_path):
    engine, TestingSessionLocal = setup_in_memory_db()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Seed minimal data
    with TestingSessionLocal() as db:
        db.add(
            models.TravelRecord(
                date=date(2024, 1, 1),
                source="A",
                destination="B",
                distance_km=10.0,
                congestion_index=0.2,
                time_of_day="08:30",
                weather="Clear",
                day_of_week="Monday",
                festival=False,
                actual_travel_time_min=12.5,
                road_type="urban",
            )
        )
        db.commit()

    # Monkeypatch model path so we don't overwrite the real artifact
    orig_model_path = predict.MODEL_PATH
    orig_model = predict.model
    predict.MODEL_PATH = os.path.join(tmp_path, "traffic_model_test.joblib")
    predict.model = None

    app.dependency_overrides[predict.get_db] = override_get_db
    client = TestClient(app)

    try:
        resp = client.post("/predictions/train")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["records_used"] == 1
        assert os.path.exists(predict.MODEL_PATH)
        assert predict.model is not None
    finally:
        # Restore globals and overrides
        predict.MODEL_PATH = orig_model_path
        predict.model = orig_model
        predict._load_model()
        app.dependency_overrides.pop(predict.get_db, None)


def test_train_endpoint_without_data_returns_400(tmp_path):
    engine, TestingSessionLocal = setup_in_memory_db()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    orig_model_path = predict.MODEL_PATH
    orig_model = predict.model
    predict.MODEL_PATH = os.path.join(tmp_path, "traffic_model_test.joblib")
    predict.model = None

    app.dependency_overrides[predict.get_db] = override_get_db
    client = TestClient(app)

    try:
        resp = client.post("/predictions/train")
        assert resp.status_code == 400, resp.text
        data = resp.json()
        assert "No data available" in data["detail"]
    finally:
        predict.MODEL_PATH = orig_model_path
        predict.model = orig_model
        predict._load_model()
        app.dependency_overrides.pop(predict.get_db, None)


def test_train_multi_and_predict(tmp_path):
    engine, TestingSessionLocal = setup_in_memory_db()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    with TestingSessionLocal() as db:
        db.add(
            models.TravelRecord(
                date=date(2024, 1, 1),
                source="A",
                destination="B",
                distance_km=10.0,
                congestion_index=0.2,
                time_of_day="08:30",
                weather="Clear",
                day_of_week="Monday",
                festival=False,
                actual_travel_time_min=12.5,
                road_type="urban",
            )
        )
        db.commit()

    # Monkeypatch paths
    orig_model_multi_path = predict.MODEL_PATH_MULTI
    orig_model_multi = predict.model_multi
    predict.MODEL_PATH_MULTI = os.path.join(tmp_path, "traffic_model_multi_test.joblib")
    predict.model_multi = None

    app.dependency_overrides[predict.get_db] = override_get_db
    client = TestClient(app)

    try:
        train_resp = client.post("/predictions/train-multi")
        assert train_resp.status_code == 200, train_resp.text
        assert os.path.exists(predict.MODEL_PATH_MULTI)
        assert predict.model_multi is not None

        predict_payload = {
            "time_of_day": "09:15",
            "weather": "Clear",
            "day_of_week": "Tuesday",
            "festival": False,
            "road_type": "urban",
            "source": "A",
            "destination": "B",
        }
        predict_resp = client.post("/predictions/multi", json=predict_payload)
        assert predict_resp.status_code == 200, predict_resp.text
        result = predict_resp.json()
        assert set(result.keys()) == {"distance_km", "congestion_index", "actual_travel_time_min"}
    finally:
        predict.MODEL_PATH_MULTI = orig_model_multi_path
        predict.model_multi = orig_model_multi
        predict._load_model_multi()
        app.dependency_overrides.pop(predict.get_db, None)


def test_predict_multi_rejects_unknown_route(tmp_path):
    engine, TestingSessionLocal = setup_in_memory_db()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    with TestingSessionLocal() as db:
        db.add(
            models.TravelRecord(
                date=date(2024, 1, 1),
                source="KnownA",
                destination="KnownB",
                distance_km=5.0,
                congestion_index=0.1,
                time_of_day="08:00",
                weather="Clear",
                day_of_week="Monday",
                festival=False,
                actual_travel_time_min=8.0,
                road_type="urban",
            )
        )
        db.commit()

    orig_model_multi_path = predict.MODEL_PATH_MULTI
    orig_model_multi = predict.model_multi
    predict.MODEL_PATH_MULTI = os.path.join(tmp_path, "traffic_model_multi_test.joblib")
    predict.model_multi = None

    app.dependency_overrides[predict.get_db] = override_get_db
    client = TestClient(app)

    try:
        train_resp = client.post("/predictions/train-multi")
        assert train_resp.status_code == 200, train_resp.text

        bad_payload = {
            "time_of_day": "09:00",
            "weather": "Clear",
            "day_of_week": "Tuesday",
            "festival": False,
            "road_type": "urban",
            "source": "UnknownA",
            "destination": "KnownB",
        }
        predict_resp = client.post("/predictions/multi", json=bad_payload)
        assert predict_resp.status_code == 400
        assert "Unknown source" in predict_resp.json()["detail"]
    finally:
        predict.MODEL_PATH_MULTI = orig_model_multi_path
        predict.model_multi = orig_model_multi
        predict._load_model_multi()
        app.dependency_overrides.pop(predict.get_db, None)
