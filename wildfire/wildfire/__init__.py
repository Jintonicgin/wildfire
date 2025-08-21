from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from config import Config
from wildfire.dataset.model_definitions import EnsembleClassifier, EnsembleRegressor
from dotenv import load_dotenv
import os

load_dotenv()

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
    from .views import main_views, auth_views, agent_views

    app.register_blueprint(main_views.bp)
    app.register_blueprint(auth_views.bp)
    app.register_blueprint(agent_views.bp)

    return app

