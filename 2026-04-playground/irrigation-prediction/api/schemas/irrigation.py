from pydantic import BaseModel, Field
from typing import Optional


class IrrigationRequest(BaseModel):
    # Numerical features
    temperature: float = Field(..., description="Temperature in celsius")
    humidity: float = Field(..., description="Relative humidity percentage")
    soil_moisture: float = Field(..., description="Soil moisture level")
    wind_speed: float = Field(..., description="Wind speed in km/h")
    solar_radiation: float = Field(..., description="Solar radiation in W/m2")
    
    # Categorical features
    crop_type: str = Field(..., description="Type of crop")
    soil_type: str = Field(..., description="Type of soil")

    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 28.5,
                "humidity": 65.0,
                "soil_moisture": 0.3,
                "wind_speed": 12.0,
                "solar_radiation": 450.0,
                "crop_type": "Wheat",
                "soil_type": "Loam"
            }
        }


class IrrigationResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted irrigation need: Low, Medium, or High")
    predicted_label: int = Field(..., description="Encoded label: 0, 1, or 2")
    model_version: str = Field(..., description="Model version used for prediction")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str