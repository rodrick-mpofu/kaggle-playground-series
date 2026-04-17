from pydantic import BaseModel, Field


class IrrigationRequest(BaseModel):
    # Numerical features
    Soil_pH: float = Field(..., description="Soil pH level")
    Soil_Moisture: float = Field(..., description="Soil moisture level")
    Organic_Carbon: float = Field(..., description="Organic carbon content")
    Electrical_Conductivity: float = Field(..., description="Electrical conductivity of soil")
    Temperature_C: float = Field(..., description="Temperature in Celsius")
    Humidity: float = Field(..., description="Relative humidity percentage")
    Rainfall_mm: float = Field(..., description="Rainfall in millimeters")
    Sunlight_Hours: float = Field(..., description="Daily sunlight hours")
    Wind_Speed_kmh: float = Field(..., description="Wind speed in km/h")
    Field_Area_hectare: float = Field(..., description="Field area in hectares")
    Previous_Irrigation_mm: float = Field(..., description="Previous irrigation amount in mm")

    # Categorical features
    Soil_Type: str = Field(..., description="Type of soil")
    Crop_Type: str = Field(..., description="Type of crop")
    Crop_Growth_Stage: str = Field(..., description="Current growth stage of the crop")
    Season: str = Field(..., description="Current season")
    Irrigation_Type: str = Field(..., description="Type of irrigation system")
    Water_Source: str = Field(..., description="Source of water")
    Mulching_Used: str = Field(..., description="Whether mulching is used")
    Region: str = Field(..., description="Geographic region")

    class Config:
        json_schema_extra = {
            "example": {
                "Soil_pH": 6.5,
                "Soil_Moisture": 0.35,
                "Organic_Carbon": 1.2,
                "Electrical_Conductivity": 0.8,
                "Temperature_C": 28.5,
                "Humidity": 65.0,
                "Rainfall_mm": 12.0,
                "Sunlight_Hours": 7.5,
                "Wind_Speed_kmh": 15.0,
                "Field_Area_hectare": 2.5,
                "Previous_Irrigation_mm": 20.0,
                "Soil_Type": "Loam",
                "Crop_Type": "Wheat",
                "Crop_Growth_Stage": "Vegetative",
                "Season": "Summer",
                "Irrigation_Type": "Drip",
                "Water_Source": "Groundwater",
                "Mulching_Used": "Yes",
                "Region": "North"
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