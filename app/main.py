from fastapi import FastAPI
from app.api.routes import router
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Colon Cancer Classification API")

Instrumentator().instrument(app).expose(app)
print("After including router")

app.include_router(router)

@app.get("/")
def home():
    return {"message":"API is running"}

