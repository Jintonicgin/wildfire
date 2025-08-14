from flask import Blueprint, render_template, request, jsonify
from wildfire import db
from wildfire.models import RegionFeature
from wildfire.dataset.predict import predict_simulation
from wildfire.dataset.predict_from_feature import predict_from_features
from wildfire.dataset.model_definitions import EnsembleClassifier, EnsembleRegressor
import datetime
import math
import sys
import os

if '__main__' in sys.modules:
    setattr(sys.modules['__main__'], 'EnsembleClassifier', EnsembleClassifier)
    setattr(sys.modules['__main__'], 'EnsembleRegressor', EnsembleRegressor)

bp = Blueprint('main', __name__)

def convert_degree_to_direction(deg):
    if deg is None or not isinstance(deg, (int, float)) or deg == -999:
        return 'N'
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = int(((deg + 22.5) % 360) / 45)
    return dirs[index]

def calculate_distance(lat1, lon1, lat2, lon2):
    if lat1 == -999 or lon1 == -999 or lat2 == -999 or lon2 == -999:
        return 0
    R = 6371e3  # metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2) * math.sin(delta_phi / 2) +
        math.cos(phi1) * math.cos(phi2) *
        math.sin(delta_lambda / 2) * math.sin(delta_lambda / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@bp.route('/')
def index():
    return render_template('nav_page/main.html')

@bp.route('/prediction')
def prediction():
    return render_template('nav_page/prediction.html')

@bp.route('/aboutus')
def aboutus():
    return render_template('nav_page/aboutus.html')

@bp.route('/faq')
def faq():
    return render_template('nav_page/faq.html')

@bp.route('/predict', methods=['POST'])
def predict_wildfire():
    data = request.get_json()

    city_name = data.get('city_name')
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    duration_hours = data.get('durationHours')
    timestamp_str = data.get('timestamp')

    if not duration_hours:
        return jsonify({"error": "예측 시간을 선택해주세요."}), 400

    try:
        timestamp = datetime.datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.datetime.utcnow()
        prediction_result = {}
        initial_lat = None
        initial_lon = None

        if city_name:
            region_feature = RegionFeature.query.filter_by(region_name=city_name).first()
            if not region_feature:
                return jsonify({"error": f"'{city_name}'에 대한 지역 피처를 찾을 수 없습니다."}), 404

            initial_lat = region_feature.lat
            initial_lon = region_feature.lng

            # Prepare input for predict_from_features
            input_for_predict_from_features = {
                "t2m": region_feature.t2m,
                "rh2m": region_feature.rh2m,
                "ws10m": region_feature.ws10m,
                "wd10m": region_feature.wd10m,
                "prectotcorr": region_feature.prectotcorr,
                "ps": region_feature.ps,
                "allsky_sfc_sw_dwn": region_feature.allsky_sfc_sw_dwn,
                "elevation_mean": region_feature.elevation_mean,
                "elevation_min": region_feature.elevation_min,
                "elevation_max": region_feature.elevation_max,
                "elevation_std": region_feature.elevation_std,
                "slope_mean": region_feature.slope_mean,
                "slope_min": region_feature.slope_min,
                "slope_max": region_feature.slope_max,
                "slope_std": region_feature.slope_std,
                "aspect_mode": region_feature.aspect_mode,
                "aspect_std": region_feature.aspect_std,
                "ndvi_before": region_feature.ndvi_before,
                "treecover_pre_fire_5x5": region_feature.treecover_pre_fire_5x5,
                "ffmc": region_feature.ffmc,
                "dmc": region_feature.dmc,
                "dc": region_feature.dc,
                "isi": region_feature.isi,
                "bui": region_feature.bui,
                "fwi": region_feature.fwi,
                "dry_windy_combo": region_feature.dry_windy_combo,
                "fuel_combo": region_feature.fuel_combo,
                "potential_spread_index": region_feature.potential_spread_index,
                "terrain_var_effect": region_feature.terrain_var_effect,
                "wind_steady_flag": region_feature.wind_steady_flag,
                "dry_to_rain_ratio_30d": region_feature.dry_to_rain_ratio_30d,
                "ndvi_stress": region_feature.ndvi_stress,
                "is_spring": region_feature.is_spring,
                "is_summer": region_feature.is_summer,
                "is_autumn": region_feature.is_autumn,
                "is_winter": region_feature.is_winter,
                "lat": initial_lat,
                "lng": initial_lon,
                "durationhours": duration_hours,
            }
            prediction_result = predict_from_features(input_for_predict_from_features)

        elif latitude is not None and longitude is not None:
            initial_lat = latitude
            initial_lon = longitude
            input_for_predict_simulation = {
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": timestamp.isoformat(),
                "durationHours": duration_hours
            }
            prediction_result = predict_simulation(input_for_predict_simulation)
        else:
            return jsonify({"error": "지역을 선택하거나 위경도를 입력해주세요."}), 400

        if "error" in prediction_result:
            return jsonify(prediction_result), 500
        
        final_damage_area = prediction_result.get("final_damage_area", 0)
        path_trace = prediction_result.get("path_trace", [])
        
        spread_direction = "N/A"
        spread_speed_value = 0.0
        total_distance = 0.0

        if city_name: # Case: predict_from_features
            area_m2 = final_damage_area * 10000
            aspect_ratio = 0.6 # From prediction.js
            b = math.sqrt(area_m2 / (math.pi * aspect_ratio))
            a = b / aspect_ratio # This 'a' is the totalDistance for ellipse
            total_distance = a
            
            if path_trace:
                last_trace = path_trace[-1]
                wind_direction_deg = last_trace.get("wind_direction_deg")
                spread_direction = convert_degree_to_direction(wind_direction_deg)
            else:
                spread_direction = "N/A"

            if duration_hours > 0:
                spread_speed_value = total_distance / duration_hours
            else:
                spread_speed_value = 0.0

        elif latitude is not None and longitude is not None: # Case: predict_simulation
            final_lat = prediction_result.get("final_lat")
            final_lon = prediction_result.get("final_lon")
            
            if initial_lat is not None and initial_lon is not None and final_lat is not None and final_lon is not None:
                total_distance = calculate_distance(initial_lat, initial_lon, final_lat, final_lon)
                if duration_hours > 0:
                    spread_speed_value = total_distance / duration_hours
                else:
                    spread_speed_value = 0.0
            else:
                total_distance = 0.0
                spread_speed_value = 0.0

            if path_trace:
                last_trace = path_trace[-1]
                wind_direction_deg = last_trace.get("wind_direction_deg")
                spread_direction = convert_degree_to_direction(wind_direction_deg)
            else:
                spread_direction = "N/A"

        spread_speed = f"{spread_speed_value:.2f} m/h" # Format for display

        return jsonify({
            "final_damage_area": final_damage_area,
            "predicted_spread_direction": spread_direction,
            "predicted_spread_speed": spread_speed,
            "total_spread_distance": total_distance, # Add this to return
            "path_trace": path_trace
        })

    except Exception as e:
        import traceback
        return jsonify({"error": f"서버 내부 오류: {str(e)}", "traceback": traceback.format_exc()}), 500
