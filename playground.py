from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sqlalchemy import create_engine

# DB connection
engine = create_engine("postgresql://thamjid:thachu#01@localhost:5432/traffic_ai")

# Load data
df = pd.read_sql("SELECT * FROM travel_records", engine)

# Encode categorical columns
df = pd.get_dummies(df, columns=["time_of_day", "weather", "road_type", "day_of_week", "festival"], drop_first=True)

# Features & target
X = df.drop(columns=["id", "date", "source", "destination", "actual_travel_time_min"])
y = df["actual_travel_time_min"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Optional: test
preds = model.predict(X_test)

print("Sample Predictions:", preds[:5])