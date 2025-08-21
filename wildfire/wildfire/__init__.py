from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from config import Config
from wildfire.dataset.model_definitions import EnsembleClassifier, EnsembleRegressor
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["KAKAO_MAP_KEY"] = os.getenv("KAKAO_MAP_KEY")

    @app.context_processor
    def inject_public_keys():
        return {"kakao_key": app.config.get("KAKAO_MAP_KEY")}

    db.init_app(app)
    migrate.init_app(app,db)

    from . import models
    from .views import main_views, auth_views, gai_views

    app.register_blueprint(main_views.bp)
    app.register_blueprint(auth_views.bp)
    app.register_blueprint(gai_views.bp)

    # Load AI models at startup for cloud environment
    with app.app_context():
        gai_views.load_pipelines()

    return app

