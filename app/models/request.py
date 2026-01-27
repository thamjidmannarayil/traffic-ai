from pydantic import BaseModel
from datetime import date
from typing import Optional

class TravelRecordCreate(BaseModel):
	date: date
	source: str
	destination: str
	distance_km: float
	congestion_index: float
	time_of_day: str
	weather: str
	day_of_week: str
	festival: bool
	actual_travel_time_min: float
	road_type: str

class TravelRecordUpdate(BaseModel):
	date: Optional[date] = None
	source: Optional[str] = None
	destination: Optional[str] = None
	distance_km: Optional[float] = None
	congestion_index: Optional[float] = None
	time_of_day: Optional[str] = None
	weather: Optional[str] = None
	day_of_week: Optional[str] = None
	festival: Optional[bool] = None
	actual_travel_time_min: Optional[float] = None
	road_type: Optional[str] = None
