import pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routers import predict, health
from api.dependencies import model_manager


def load_artifacts():
    import os

    artifacts_path = "models/artifacts.pkl"

    if not os.path.exists(artifacts_path):
        print("WARNING: No model artifacts found at models/artifacts.pkl")
        print("Run python main.py first to train and save the model")
        return

    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    # Use the model URI saved during training
    model_uri = artifacts["model_uri"]
    print(f"Loading model from MLflow URI: {model_uri}")

    model_manager.load(
        model_uri=model_uri,
        label_encoder=artifacts["label_encoder"],
        feature_columns=artifacts["feature_columns"],
        version=artifacts.get("version", "1.0.0")
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on startup
    load_artifacts()
    yield
    # Runs on shutdown
    print("Shutting down API")


app = FastAPI(
    title="Irrigation Need Prediction API",
    description="Predicts irrigation need (Low, Medium, High) from environmental conditions",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")