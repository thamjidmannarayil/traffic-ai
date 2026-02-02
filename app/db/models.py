from sqlalchemy import Column, Integer, String, Float, Date, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TravelRecord(Base):
	__tablename__ = "travel_records"

	id = Column(Integer, primary_key=True, index=True)
	date = Column(Date, nullable=False)
	source = Column(String, nullable=False)
	destination = Column(String, nullable=False)
	distance_km = Column(Float, nullable=False)
	congestion_index = Column(Float, nullable=False)
	time_of_day = Column(String, nullable=False)
	weather = Column(String, nullable=False)
	day_of_week = Column(String, nullable=False)
	festival = Column(Boolean, nullable=False, default=False)
	actual_travel_time_min = Column(Float, nullable=False)
	road_type = Column(String, nullable=False)
