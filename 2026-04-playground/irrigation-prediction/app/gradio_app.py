import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/predict")

# Based on your actual dataset — update these with real unique values
SOIL_TYPES = ["Sandy", "Loam", "Clay", "Silt", "Peaty", "Chalky"]
CROP_TYPES = ["Wheat", "Corn", "Rice", "Soybean", "Cotton", "Sugarcane"]
CROP_GROWTH_STAGES = ["Germination", "Vegetative", "Flowering", "Maturity"]
SEASONS = ["Summer", "Winter", "Spring", "Autumn"]
IRRIGATION_TYPES = ["Drip", "Sprinkler", "Flood", "Manual"]
WATER_SOURCES = ["Groundwater", "River", "Rainwater", "Reservoir"]
MULCHING_USED = ["Yes", "No"]
REGIONS = ["North", "South", "East", "West", "Central"]


def predict_irrigation(
    soil_ph, soil_moisture, organic_carbon, electrical_conductivity,
    temperature_c, humidity, rainfall_mm, sunlight_hours,
    wind_speed_kmh, field_area_hectare, previous_irrigation_mm,
    soil_type, crop_type, crop_growth_stage, season,
    irrigation_type, water_source, mulching_used, region
):
    payload = {
        "Soil_pH": soil_ph,
        "Soil_Moisture": soil_moisture,
        "Organic_Carbon": organic_carbon,
        "Electrical_Conductivity": electrical_conductivity,
        "Temperature_C": temperature_c,
        "Humidity": humidity,
        "Rainfall_mm": rainfall_mm,
        "Sunlight_Hours": sunlight_hours,
        "Wind_Speed_kmh": wind_speed_kmh,
        "Field_Area_hectare": field_area_hectare,
        "Previous_Irrigation_mm": previous_irrigation_mm,
        "Soil_Type": soil_type,
        "Crop_Type": crop_type,
        "Crop_Growth_Stage": crop_growth_stage,
        "Season": season,
        "Irrigation_Type": irrigation_type,
        "Water_Source": water_source,
        "Mulching_Used": mulching_used,
        "Region": region
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        label_colors = {
            "Low": "🟢 Low",
            "Medium": "🟡 Medium",
            "High": "🔴 High"
        }

        predicted = result["predicted_class"]
        version = result["model_version"]

        return (
            label_colors.get(predicted, predicted),
            f"Model version: {version}"
        )

    except requests.exceptions.ConnectionError:
        return "❌ Error", "API is not running. Start it with: uvicorn api.main:app --reload"
    except Exception as e:
        return "❌ Error", str(e)


with gr.Blocks(title="Irrigation Need Predictor", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🌱 Irrigation Need Predictor
    Predict whether irrigation need is **Low**, **Medium**, or **High** based on environmental conditions.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🌡️ Environmental Conditions")
            temperature_c = gr.Slider(0, 50, value=28.5, label="Temperature (°C)")
            humidity = gr.Slider(0, 100, value=65.0, label="Humidity (%)")
            rainfall_mm = gr.Slider(0, 300, value=12.0, label="Rainfall (mm)")
            sunlight_hours = gr.Slider(0, 16, value=7.5, label="Sunlight Hours")
            wind_speed_kmh = gr.Slider(0, 100, value=15.0, label="Wind Speed (km/h)")

        with gr.Column():
            gr.Markdown("### 🪨 Soil Properties")
            soil_ph = gr.Slider(0, 14, value=6.5, label="Soil pH")
            soil_moisture = gr.Slider(0, 1, value=0.35, label="Soil Moisture")
            organic_carbon = gr.Slider(0, 10, value=1.2, label="Organic Carbon")
            electrical_conductivity = gr.Slider(0, 5, value=0.8, label="Electrical Conductivity")

        with gr.Column():
            gr.Markdown("### 🌾 Crop & Field Info")
            crop_type = gr.Dropdown(CROP_TYPES, value="Wheat", label="Crop Type")
            crop_growth_stage = gr.Dropdown(CROP_GROWTH_STAGES, value="Vegetative", label="Crop Growth Stage")
            field_area_hectare = gr.Slider(0, 100, value=2.5, label="Field Area (hectares)")
            previous_irrigation_mm = gr.Slider(0, 100, value=20.0, label="Previous Irrigation (mm)")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🗂️ Categorical Info")
            soil_type = gr.Dropdown(SOIL_TYPES, value="Loam", label="Soil Type")
            season = gr.Dropdown(SEASONS, value="Summer", label="Season")
            irrigation_type = gr.Dropdown(IRRIGATION_TYPES, value="Drip", label="Irrigation Type")
            water_source = gr.Dropdown(WATER_SOURCES, value="Groundwater", label="Water Source")
            mulching_used = gr.Dropdown(MULCHING_USED, value="Yes", label="Mulching Used")
            region = gr.Dropdown(REGIONS, value="North", label="Region")

    with gr.Row():
        predict_btn = gr.Button("🔍 Predict Irrigation Need", variant="primary", scale=1)

    with gr.Row():
        prediction_output = gr.Text(label="Predicted Irrigation Need", interactive=False)
        model_info = gr.Text(label="Model Info", interactive=False)

    predict_btn.click(
        fn=predict_irrigation,
        inputs=[
            soil_ph, soil_moisture, organic_carbon, electrical_conductivity,
            temperature_c, humidity, rainfall_mm, sunlight_hours,
            wind_speed_kmh, field_area_hectare, previous_irrigation_mm,
            soil_type, crop_type, crop_growth_stage, season,
            irrigation_type, water_source, mulching_used, region
        ],
        outputs=[prediction_output, model_info]
    )

    gr.Markdown("""
    ---
    > **Note:** Make sure the FastAPI server is running before predicting.  
    > Start it with: `uvicorn api.main:app --reload`
    """)

if __name__ == "__main__":
    demo.launch()