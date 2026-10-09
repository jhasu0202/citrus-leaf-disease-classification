from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Citrus Leaf Disease Classification API",
    description=(
        "Classify citrus leaf images using seven trained "
        "deep-learning architectures."
    ),
    version="1.1.0",
)

app.include_router(router)


@app.get("/")
def home():
    return {
        "project": "Citrus Leaf Disease Classification",
        "status": "running",
        "docs": "/docs",
        "models": "/models",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "running"}