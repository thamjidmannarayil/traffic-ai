from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.models import TravelRecord
from app.db.session import SessionLocal
from app.models.request import TravelRecordCreate, TravelRecordUpdate
from app.models.response import TravelRecordOut

router = APIRouter(prefix="/records", tags=["Travel Records"])

def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()

# Create
@router.post("/", response_model=TravelRecordOut)
def create_record(record: TravelRecordCreate, db: Session = Depends(get_db)):
	db_record = TravelRecord(**record.dict())
	db.add(db_record)
	db.commit()
	db.refresh(db_record)
	return db_record

# Read all
@router.get("/", response_model=List[TravelRecordOut])
def read_records(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
	return db.query(TravelRecord).offset(skip).limit(limit).all()

# Read one
@router.get("/{record_id}", response_model=TravelRecordOut)
def read_record(record_id: int, db: Session = Depends(get_db)):
	record = db.query(TravelRecord).filter(TravelRecord.id == record_id).first()
	if not record:
		raise HTTPException(status_code=404, detail="Record not found")
	return record

# Update
@router.put("/{record_id}", response_model=TravelRecordOut)
def update_record(record_id: int, record_update: TravelRecordUpdate, db: Session = Depends(get_db)):
	record = db.query(TravelRecord).filter(TravelRecord.id == record_id).first()
	if not record:
		raise HTTPException(status_code=404, detail="Record not found")
	for field, value in record_update.dict(exclude_unset=True).items():
		setattr(record, field, value)
	db.commit()
	db.refresh(record)
	return record

# Delete
@router.delete("/{record_id}", response_model=dict)
def delete_record(record_id: int, db: Session = Depends(get_db)):
	record = db.query(TravelRecord).filter(TravelRecord.id == record_id).first()
	if not record:
		raise HTTPException(status_code=404, detail="Record not found")
	db.delete(record)
	db.commit()
	return {"ok": True}
