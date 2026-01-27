from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import csv
import datetime
from app.db.models import TravelRecord
from app.db.session import SessionLocal
from io import StringIO

router = APIRouter(prefix="/api/v1", tags=["Export Records"])

CSV_HEADERS = [
    "date",
    "source",
    "destination",
    "distance_km",
    "congestion_index",
    "time_of_day",
    "weather",
    "day_of_week",
    "festival",
    "actual_travel_time_min",
    "road_type"
]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/export", response_model=dict)
def export_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    content = file.file.read().decode('utf-8')
    reader = csv.DictReader(StringIO(content))
    if reader.fieldnames is None or set(CSV_HEADERS) != set(reader.fieldnames):
        raise HTTPException(status_code=400, detail=f"CSV must have headers: {', '.join(CSV_HEADERS)}")
    created = 0
    for row in reader:
        try:
            record_data = {k: row[k] for k in CSV_HEADERS}
            # Convert types as needed
            record_data['distance_km'] = float(record_data['distance_km'])
            record_data['congestion_index'] = float(record_data['congestion_index'])
            record_data['festival'] = record_data['festival'].lower() in ['true', '1', 'yes']
            record_data['actual_travel_time_min'] = float(record_data['actual_travel_time_min'])
            # Convert date from DD-MM-YYYY to YYYY-MM-DD
            try:
                day, month, year = record_data['date'].split('-')
                record_data['date'] = datetime.date(int(year), int(month), int(day))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid date format in row: {row}. Error: {str(e)}")
            # date is handled by Pydantic
            record = TravelRecord(**record_data)
            db.add(record)
            created += 1
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error in row: {row}. Error: {str(e)}")
    db.commit()
    return {"created": created}
