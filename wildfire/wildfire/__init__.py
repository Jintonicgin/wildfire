from flask import Flask, g, session
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from config import Config
from wildfire.ML.model_definitions import EnsembleClassifier, EnsembleRegressor
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
    app.config["JSON_AS_ASCII"] = False

    @app.context_processor
    def inject_public_keys():
        return {"kakao_key": app.config.get("KAKAO_MAP_KEY")}

    db.init_app(app)
    migrate.init_app(app, db)

    @app.before_request
    def load_current_user():
        g.user = None
        username = session.get("user_username")
        if username:
            from .models import Member
            g.user = db.session.get(Member, username)

    from .views import main_views, auth_views, gai_views, rag_views, vision_views

    app.register_blueprint(main_views.bp)
    app.register_blueprint(auth_views.bp)
    app.register_blueprint(gai_views.bp)
    app.register_blueprint(rag_views.bp)
    app.register_blueprint(vision_views.bp)

    return app