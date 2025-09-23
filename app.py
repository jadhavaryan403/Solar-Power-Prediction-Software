from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],  # You can specify frontend domain here instead of ""
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
with open("model_ac_compressed.joblib", "rb") as f:
    model_ac = joblib.load(f)

with open("model_dc_compressed.joblib", "rb") as f:
    model_dc = joblib.load(f)


@app.get("/")
async def home():
    return {"message": "Welcome to the page !!!"}

class UserInput(BaseModel):
    lat: float
    lon: float
    kwp: float

@app.post("/fetch")
async def fetch(input : UserInput):
    global hourly_dataframe

    lat = input.lat
    lon = input.lon
    system_kwp = input.kwp

    # Validate values
    if lat is None or lon is None or system_kwp is None:
        return JSONResponse(status_code=400, content={"error": "Missing lat, lon, or kwp in request."})
    
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "hourly": ["temperature_2m", "wind_speed_10m", "shortwave_radiation"],
        "forecast_days": 7,
        "wind_speed_unit": "ms",
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
    radiation = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert("Asia/Kolkata"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert("Asia/Kolkata"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": temperature_2m,
        "wind_speed_10m": wind_speed_10m,
        "shortwave_radiation": radiation,
    }

    hourly_dataframe = pd.DataFrame(data=hourly_data)

    if hourly_dataframe is None:
        return JSONResponse(status_code=400, content={"error": "Weather data not fetched yet."})

    # Prepare features
    features_ac = hourly_dataframe.iloc[:, 1:4].values
    ac_hourly = model_ac.predict(features_ac)

    # Prepare features for DC model
    feature_dc = np.concatenate((features_ac, ac_hourly.reshape(-1, 1)) ,axis=1)
    dc_hourly = model_dc.predict(feature_dc)

    # Set AC/DC output to 0 when radiation is 0
    for i in range(len(ac_hourly)):
        if features_ac[i][2] == 0:
            ac_hourly[i] = 0
    for i in range(len(dc_hourly)):
        if features_ac[i][2] == 0:
            dc_hourly[i] = 0

    ac_hourly = ac_hourly*(system_kwp/350)
    dc_hourly = dc_hourly*(system_kwp/350)

    return {
        "ac_hourly": ac_hourly.tolist(),
        "dc_hourly": dc_hourly.tolist()
    }