import torch

from app.core.model_loader import (
    DEVICE,
    CLASS_NAMES,
    MODEL_CONFIGS,
    get_model,
)
from app.utils.image_processing import preprocess_image


def predict_image(image, model_name="ViT-B/16"):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )

    model = get_model(model_name)
    tensor = preprocess_image(image).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax())

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index]) * 100, 2
        ),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        },
        "model": model_name,
    }