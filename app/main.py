# FastAPI app initialization
from fastapi import FastAPI

from app.api.routes import predict, export

app = FastAPI(title="Traffic API")


# Import and include routers
app.include_router(predict.router)
app.include_router(export.router)

@app.get("/")
def root():
	return {"message": "Traffic API is running"}
