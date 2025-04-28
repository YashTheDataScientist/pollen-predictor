from flask import Flask, request, jsonify
import pandas as pd
import joblib
import requests
import os

# Initialize Flask app
app = Flask(__name__)

# Load your model
model = joblib.load('pollen_risk_model.pkl')

# Load your suburb density CSV
suburb_master_df = pd.read_csv('suburb_plant_density.csv')
suburb_master_df['suburb'] = suburb_master_df['suburb'].str.strip().str.lower()

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    suburb_name = data.get('suburb')
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    if not suburb_name or latitude is None or longitude is None:
        return jsonify({'error': 'Missing suburb, latitude or longitude'}), 400

    # Fetch live weather from Open-Meteo
    weather_params = ",".join([
        'temperature_2m', 'dew_point_2m', 'relative_humidity_2m',
        'wind_speed_10m', 'cloud_cover', 'surface_pressure'
    ])
    weather_url = "https://api.open-meteo.com/v1/forecast"

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'current_weather': True,
        'hourly': weather_params,
        'timezone': 'auto'
    }

    try:
        response = requests.get(weather_url, params=params)
        weather_data = response.json()
    except Exception as e:
        return jsonify({'error': f'Weather API failed: {str(e)}'}), 500

    # Extract necessary weather features
    current_weather = weather_data.get('current_weather', {})
    if not current_weather:
        return jsonify({'error': 'No weather data found'}), 500

    temp = current_weather.get('temperature')
    wind_speed = current_weather.get('windspeed')

    # Find plant density for the suburb
    matched = suburb_master_df[suburb_master_df['suburb'] == suburb_name.strip().lower()]
    if matched.empty:
        plant_density = 0
    else:
        plant_density = matched.iloc[0]['local_density_score']

    # Make prediction
    features = [[temp, wind_speed, plant_density] + [0] * 7]  # padding to 10 features
    prediction = model.predict(features)[0]

    return jsonify({'predicted_pollen_risk': int(prediction)})

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
