# Citrus Leaf Disease Classification

Deep-learning project for classifying citrus leaf images into five categories.

## Disease Classes
- Black spot
- Canker
- Greening
- Healthy
- Melanose

## Models
- ResNet34
- EfficientNet-B0
- Vision Transformer (ViT-B/16)
- MobileNetV3-Large
- ConvNeXt-Tiny
- Swin-Tiny
- MobileViT-Small

## Dataset
The project uses a custom citrus leaf disease dataset.
The dataset is not included in this repository.

## Run Locally

Install dependencies:

    python -m pip install -r requirements.txt

Start the API:

    python -m uvicorn app.main:app --reload

Open Swagger documentation:

    http://127.0.0.1:8000/docs

## API Endpoints
- GET /
- GET /health
- GET /classes
- GET /models
- GET /model-info
- POST /predict

## Prediction
The prediction endpoint accepts an image and an optional model name.

Available model names:
- ResNet34
- EfficientNet-B0
- ViT-B/16
- MobileNetV3-Large
- ConvNeXt-Tiny
- Swin-Tiny
- MobileViT-Small

Example:

    POST /predict?model=ViT-B%2F16

Upload the image using the file field in Swagger.

## Notes
- The trained checkpoints are stored using Git LFS.
- Predictions depend on correct checkpoint architecture, class mapping, and image preprocessing.
- Reported model accuracy is based on the project's recorded evaluation results.
- This project is for educational and research purposes.
