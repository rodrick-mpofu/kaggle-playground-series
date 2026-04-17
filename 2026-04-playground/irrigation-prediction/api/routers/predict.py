import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from api.schemas.irrigation import IrrigationRequest, IrrigationResponse
from api.dependencies import ModelManager, get_model_manager

router = APIRouter()


def request_to_dataframe(request: IrrigationRequest, feature_columns: list) -> pd.DataFrame:
    """Convert incoming request to a dataframe aligned to training columns."""
    raw = {
        "Soil_pH": request.Soil_pH,
        "Soil_Moisture": request.Soil_Moisture,
        "Organic_Carbon": request.Organic_Carbon,
        "Electrical_Conductivity": request.Electrical_Conductivity,
        "Temperature_C": request.Temperature_C,
        "Humidity": request.Humidity,
        "Rainfall_mm": request.Rainfall_mm,
        "Sunlight_Hours": request.Sunlight_Hours,
        "Wind_Speed_kmh": request.Wind_Speed_kmh,
        "Field_Area_hectare": request.Field_Area_hectare,
        "Previous_Irrigation_mm": request.Previous_Irrigation_mm,
        "Soil_Type": request.Soil_Type,
        "Crop_Type": request.Crop_Type,
        "Crop_Growth_Stage": request.Crop_Growth_Stage,
        "Season": request.Season,
        "Irrigation_Type": request.Irrigation_Type,
        "Water_Source": request.Water_Source,
        "Mulching_Used": request.Mulching_Used,
        "Region": request.Region,
    }

    df = pd.DataFrame([raw])

    # One-hot encode categoricals to match training
    df = pd.get_dummies(df, drop_first=True)

    # Align to training feature columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Convert bool cols to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


@router.post("/predict", response_model=IrrigationResponse, tags=["Prediction"])
def predict(
    request: IrrigationRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    if not manager.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first with python main.py"
        )

    try:
        input_df = request_to_dataframe(request, manager.feature_columns)
        label, class_name = manager.predict(input_df)

        return IrrigationResponse(
            predicted_class=class_name,
            predicted_label=label,
            model_version=manager.model_version
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")