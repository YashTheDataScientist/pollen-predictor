from flask import Flask, request, jsonify
import pandas as pd
import joblib
import requests
from datetime import datetime
import os

app = Flask(__name__)

# Load the trained balanced model
model = joblib.load('pollen_risk_model_balanced_new.pkl')

# Load suburb data with lat/lon and density info
suburb_df = pd.read_csv('suburb_plant_density.csv')
suburb_df['suburb'] = suburb_df['suburb'].str.strip().str.lower()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    suburb_input = data.get('suburb', '').strip().lower()

    # Validate suburb
    matched = suburb_df[suburb_df['suburb'] == suburb_input]
    if matched.empty:
        return jsonify({'error': f"Suburb '{suburb_input}' not found"}), 404

    lat = matched.iloc[0]['latitude']
    lon = matched.iloc[0]['longitude']

    # Open-Meteo API call
    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,dew_point_2m,relative_humidity_2m,cloud_cover,wind_speed_10m',
        'current_weather': True,
        'timezone': 'auto'
    }

    try:
        response = requests.get(weather_url, params=params)
        weather_data = response.json()
    except Exception as e:
        return jsonify({'error': f'Weather API failed: {str(e)}'}), 500

    current = weather_data.get('current_weather', {})
    hourly = weather_data.get('hourly', {})

    if not current or not hourly:
        return jsonify({'error': 'Missing weather data'}), 500

    timestamp = current.get('time')
    try:
        idx = hourly['time'].index(timestamp)
        temp = hourly['temperature_2m'][idx]
        dewpoint = hourly['dew_point_2m'][idx]
        humidity = hourly['relative_humidity_2m'][idx]
        wind_speed = hourly['wind_speed_10m'][idx]
        cloud_cover = hourly['cloud_cover'][idx]
    except Exception as e:
        return jsonify({'error': f'Could not align timestamp: {str(e)}'}), 500

    # Prepare input for prediction
    month = datetime.now().month
    features = pd.DataFrame([{
        'temperature': temp,
        'dewpoint_temperature': dewpoint,
        'wind_speed': wind_speed,
        'relative_humidity': humidity,
        'total_cloud_cover': cloud_cover,
        'month': month
    }])

    prediction = model.predict(features)[0]

    return jsonify({
        'suburb': suburb_input.title(),
        'predicted_pollen_risk': int(prediction),
        'features_used': {
            'temperature': temp,
            'dewpoint_temperature': dewpoint,
            'relative_humidity': humidity,
            'wind_speed': wind_speed,
            'total_cloud_cover': cloud_cover,
            'month': month
        }
    })

# Run app locally
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
