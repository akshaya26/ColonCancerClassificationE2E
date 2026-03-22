from fastapi import APIRouter, UploadFile, File, HTTPException
from app.model.predict import predict
from app.schemas.response import PredictionResponse

from app.services.cache import get_cache, set_cache
from app.utils.hashing import get_hash
import time

router = APIRouter()

@router.get("/")
def home():
    return {"message": "Colon Cancer Detection API"}

@router.post("/predict", response_model = PredictionResponse)
async def predict_image(file: UploadFile=File(...)):
    try:
        image_bytes = await file.read()
        key = get_hash(image_bytes)

        cached = get_cache(key) 
        if cached:
            return cached
        
        # start = time.time()
        result = predict(image_bytes)

        set_cache(key, result)

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail = str(e))
    

