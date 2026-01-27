# FastAPI app initialization
from fastapi import FastAPI

from app.api.routes import predict

app = FastAPI(title="Traffic API")


# Import and include routers
app.include_router(predict.router)

@app.get("/")
def root():
	return {"message": "Traffic API is running"}
