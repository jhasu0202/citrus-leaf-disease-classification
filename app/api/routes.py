from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from PIL import Image, UnidentifiedImageError

from app.core.model_loader import CLASS_NAMES, MODEL_CONFIGS
from app.services.prediction_service import predict_image

router = APIRouter()


@router.get("/classes")
def get_classes():
    return {"classes": CLASS_NAMES}


@router.get("/models")
def get_models():
    return {
        "models": list(MODEL_CONFIGS.keys()),
        "default_model": "ViT-B/16",
    }


@router.get("/model-info")
def model_info():
    return {
        "models": list(MODEL_CONFIGS.keys()),
        "classes": CLASS_NAMES,
        "reported_vit_validation_accuracy": 88.33,
        "accuracy_unit": "percent",
    }


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query(
        default="ViT-B/16",
        description="Model name; see GET /models",
    ),
):
    if model not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unsupported model",
                "available_models": list(MODEL_CONFIGS.keys()),
            },
        )

    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/webp",
    ):
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )

    try:
        return predict_image(image, model)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )