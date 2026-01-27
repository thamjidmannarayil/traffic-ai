from pydantic import BaseModel
from datetime import date

class TravelRecordOut(BaseModel):
	id: int
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

	class Config:
		from_attributes = True
