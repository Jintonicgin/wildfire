from wildfire import db
from datetime import datetime

class Member(db.Model):
    __tablename__ = 'member'
    __bind_key__ = 'seed'

    username = db.Column(db.String(30), primary_key=True, unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class RegionFeature(db.Model):
    __tablename__ = 'region_prediction_features'
    __bind_key__ = 'wildfire_dataset'

    region_name = db.Column(db.String(100), primary_key=True, nullable=False)
    lat = db.Column(db.Float)
    lng = db.Column(db.Float)
    t2m = db.Column(db.Float)
    rh2m = db.Column(db.Float)
    ws10m = db.Column(db.Float)
    wd10m = db.Column(db.Float)
    prectotcorr = db.Column(db.Float)
    ps = db.Column(db.Float)
    allsky_sfc_sw_dwn = db.Column(db.Float)
    elevation_mean = db.Column(db.Float)
    elevation_min = db.Column(db.Float)
    elevation_max = db.Column(db.Float)
    elevation_std = db.Column(db.Float)
    slope_mean = db.Column(db.Float)
    slope_min = db.Column(db.Float)
    slope_max = db.Column(db.Float)
    slope_std = db.Column(db.Float)
    aspect_mode = db.Column(db.Float)
    aspect_std = db.Column(db.Float)
    ndvi_before = db.Column(db.Float)
    treecover_pre_fire_5x5 = db.Column(db.Float)
    ffmc = db.Column(db.Float)
    dmc = db.Column(db.Float)
    dc = db.Column(db.Float)
    isi = db.Column(db.Float)
    bui = db.Column(db.Float)
    fwi = db.Column(db.Float)
    dry_windy_combo = db.Column(db.Float)
    fuel_combo = db.Column(db.Float)
    potential_spread_index = db.Column(db.Float)
    terrain_var_effect = db.Column(db.Float)
    wind_steady_flag = db.Column(db.Integer) # Assuming 0 or 1
    dry_to_rain_ratio_30d = db.Column(db.Float)
    ndvi_stress = db.Column(db.Float)
    is_spring = db.Column(db.Integer) # Assuming 0 or 1
    is_summer = db.Column(db.Integer) # Assuming 0 or 1
    is_autumn = db.Column(db.Integer) # Assuming 0 or 1
    is_winter = db.Column(db.Integer) # Assuming 0 or 1
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)