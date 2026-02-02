from pydantic import BaseModel, field_validator
from datetime import date, datetime
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
	road_type: str

	@field_validator("time_of_day")
	@classmethod
	def validate_time_format(cls, value: str) -> str:
		"""
		Ensure time_of_day is provided as HH:MM (24-hour).
		This prevents downstream preprocessing failures.
		"""
		try:
			datetime.strptime(value, "%H:%M")
		except ValueError:
			raise ValueError("time_of_day must be in 'HH:MM' format")
		return value

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

	@field_validator("time_of_day")
	@classmethod
	def validate_time_format(cls, value: Optional[str]) -> Optional[str]:
		if value is None:
			return value
		try:
			datetime.strptime(value, "%H:%M")
		except ValueError:
			raise ValueError("time_of_day must be in 'HH:MM' format")
		return value


class MultiPredictRequest(BaseModel):
	date: Optional[date] = None
	source: Optional[str] = None
	destination: Optional[str] = None
	time_of_day: str
	weather: str
	day_of_week: str
	festival: bool
	road_type: str

	@field_validator("time_of_day")
	@classmethod
	def validate_time_format(cls, value: str) -> str:
		try:
			datetime.strptime(value, "%H:%M")
		except ValueError:
			raise ValueError("time_of_day must be in 'HH:MM' format")
		return value
