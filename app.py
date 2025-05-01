from flask import Flask, request, jsonify
import pandas as pd
import joblib
import requests
import os
from datetime import datetime
from flask_cors import CORS


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your trained model
model = joblib.load('pollen_risk_model_balanced_new.pkl')

# Load your suburb plant density data
suburb_master_df = pd.read_csv('suburb_plant_density.csv')
suburb_master_df['suburb'] = suburb_master_df['suburb'].str.strip().str.lower()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        suburb_name = data.get('suburb')
        latitude = data.get('latitude')
        longitude = data.get('longitude')

        if not suburb_name or latitude is None or longitude is None:
            return jsonify({'error': 'Missing suburb, latitude, or longitude'}), 400

        # Get the plant density score
        match = suburb_master_df[suburb_master_df['suburb'] == suburb_name.strip().lower()]
        plant_density = match['local_density_score'].iloc[0] if not match.empty else 0

        # Get the current month
        current_month = datetime.now().month

        # Fetch weather data from Open-Meteo
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': 'temperature_2m,dew_point_2m,relative_humidity_2m,cloud_cover,wind_speed_10m',
            'timezone': 'auto'
        }

        response = requests.get(weather_url, params=params)
        weather_data = response.json()

        if 'hourly' not in weather_data or 'temperature_2m' not in weather_data['hourly']:
            return jsonify({'error': 'Incomplete weather data'}), 500

        # Use the first (latest available) hour of forecast
        latest_idx = 0

        temperature = weather_data['hourly']['temperature_2m'][latest_idx]
        dewpoint_temperature = weather_data['hourly']['dew_point_2m'][latest_idx]
        relative_humidity = weather_data['hourly']['relative_humidity_2m'][latest_idx]
        cloud_cover = weather_data['hourly']['cloud_cover'][latest_idx]
        wind_speed = weather_data['hourly']['wind_speed_10m'][latest_idx]

        # Prepare input features (order matters!)
        features = [[
            temperature,
            dewpoint_temperature,
            wind_speed,
            relative_humidity,
            cloud_cover,
            current_month
        ]]

        # Predict
        prediction = model.predict(features)[0]

        return jsonify({
            'predicted_pollen_risk': int(prediction),
            'features_used': {
                'temperature': temperature,
                'dewpoint_temperature': dewpoint_temperature,
                'wind_speed': wind_speed,
                'relative_humidity': relative_humidity,
                'cloud_cover': cloud_cover,
                'month': current_month,
                'plant_density': plant_density
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the app locally
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
