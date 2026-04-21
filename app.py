import os
from flask import Flask, render_template, send_from_directory, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from logging.handlers import RotatingFileHandler
import secrets
from datetime import timedelta
import pymysql
from waitress import serve
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

from models import db
from routes import main_bp

pymysql.install_as_MySQLdb()

app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///summarizer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_COOKIE_SECURE'] = False  # For local testing
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Initialize SQLAlchemy
db.init_app(app)

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

# Initialize database with error handling
try:
    with app.app_context():
        # Keep existing DB intact to preserve users between restarts if possible
        db.create_all()
        app.logger.info("Database initialized successfully")
except Exception as e:
    app.logger.error(f"Failed to initialize database: {str(e)}")
    raise

# Limiter initialization
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",
    strategy="fixed-window",
    default_limits=["200 per day", "50 per hour"]
)

# Register Blueprints
app.register_blueprint(main_bp)

@app.route('/')
def home():
    app.logger.debug("Serving home page")
    return render_template('index.html')

@app.route('/google3a8738f31820d.html')
def google_verification():
    """Bulletproof Google verification endpoint"""
    try:
        response = send_from_directory(
            os.path.join(app.root_path, 'static'),
            'google3a8738f31820d.html',
            mimetype='text/plain'
        )
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        return response
    except Exception as e:
        app.logger.error(f"Static file serving failed: {str(e)}")
    
    # Fallback to direct response if file not found
    verification_content = "google-site-verification: google3a8738f31820d.html"
    try:
        return Response(
            verification_content,
            status=200,
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-store',
                'Content-Length': str(len(verification_content))
            }
        )
    except Exception as e:
        app.logger.critical(f"Verification failed completely: {str(e)}")
        return "Verification unavailable", 500

if __name__ == '__main__':
    app.logger.debug("Starting Waitress server...")
    serve(app, host='0.0.0.0', port=5000, threads=4)
