from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from api.predict import predict_irrigation
from api.health import check_health


app = FastAPI(
    title="Irrigation Prediction API",
    description="API for predicting irrigation needs based on various features.",
    version="1.0.0"
)

@app.get("/health")
def health():
    """Health check endpoint."""
    return check_health()



# === Request Data Schema ===


