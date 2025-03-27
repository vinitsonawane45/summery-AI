from flask import Flask, request, jsonify, session, render_template
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import T5ForConditionalGeneration, T5Tokenizer
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import asyncio
import aiohttp
from functools import lru_cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.storage import SQLAlchemyStorage# Updated import path
import logging
from logging.handlers import RotatingFileHandler
import secrets
from datetime import timedelta
import bleach
import pymysql
from threading import Lock

# Initialize NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

load_dotenv()
pymysql.install_as_MySQLdb()

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# Initialize Flask-Limiter with SQLAlchemy storage
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage=SQLAlchemyStorage(db.engine, table_name='rate_limits')
)

# [Rest of your code remains exactly the same...]

# User Model
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    preferences = db.Column(db.String(50), default='150')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

# Initialize database
with app.app_context():
    db.create_all()

# Initialize T5 model
model_lock = Lock()
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Text processing utilities
sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

@lru_cache(maxsize=128)
def summarize_text(text, max_length=150):
    try:
        if not text or len(text.strip()) == 0:
            raise ValueError("Empty text provided for summarization")
            
        sanitized_text = bleach.clean(text[:5000])
        if len(sanitized_text) < 20:
            raise ValueError("Text too short for summarization")
            
        input_text = "summarize: " + sanitized_text
        
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        with model_lock:
            summary_ids = model.generate(
                input_ids,
                max_length=max_length,
                min_length=min(20, max_length//2),
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            if not summary or len(summary.strip()) == 0:
                raise ValueError("Empty summary generated")
                
            return summary
            
    except Exception as e:
        app.logger.error(f"Summarization error: {str(e)}")
        raise ValueError(f"Summarization failed: {str(e)}")

async def fetch_url_content(url):
    try:
        if not url or not urlparse(url).scheme:
            raise ValueError("Invalid URL")
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml'
        }
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP Error {response.status}")
                
                content = await response.text()
                soup = BeautifulSoup(content, "html.parser")
                
                for element in soup(["script", "style", "iframe", "noscript", "header", "footer", "nav"]):
                    element.decompose()
                
                text_elements = soup.find_all(['p', 'h1', 'h2', 'h3'])
                text = ' '.join([element.get_text(separator=" ", strip=True) for element in text_elements])
                
                if not text.strip():
                    raise ValueError("No readable content found on page")
                
                return bleach.clean(text[:5000])
                
    except Exception as e:
        app.logger.error(f"URL fetch error: {str(e)}")
        raise ValueError(f"Failed to fetch URL content: {str(e)}")

def extract_text_from_pdf(pdf_file):
    try:
        if not pdf_file or not pdf_file.filename.lower().endswith('.pdf'):
            raise ValueError("Invalid file type. Only PDFs are accepted.")
            
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page in reader.pages[:5]:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
            if len(text) > 3000:
                break
                
        if not text.strip():
            raise ValueError("No readable text found in PDF")
            
        return bleach.clean(text[:5000])
        
    except Exception as e:
        app.logger.error(f"PDF extraction error: {str(e)}")
        raise ValueError(f"Failed to extract PDF text: {str(e)}")

def analyze_text(text):
    try:
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for analysis")
            
        words = word_tokenize(text)
        word_count = len(words)
        unique_words = len(set(words))
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            'word_count': word_count,
            'unique_words': unique_words,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'readability': "Basic" if avg_sentence_length < 15 else "Intermediate" if avg_sentence_length < 25 else "Advanced"
        }
        
    except Exception as e:
        app.logger.error(f"Text analysis error: {str(e)}")
        raise ValueError(f"Text analysis failed: {str(e)}")

def extract_keywords(text, top_n=10):
    try:
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for keyword extraction")
            
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in stop_words]
        freq_dist = nltk.FreqDist(words)
        keywords = [word for word, freq in freq_dist.most_common(top_n)]
        
        return keywords
        
    except Exception as e:
        app.logger.error(f"Keyword extraction error: {str(e)}")
        raise ValueError(f"Keyword extraction failed: {str(e)}")

def analyze_sentiment(text):
    try:
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for sentiment analysis")
            
        scores = sid.polarity_scores(text)
        sentiment = "Positive" if scores['compound'] > 0.05 else "Negative" if scores['compound'] < -0.05 else "Neutral"
        
        return {
            'sentiment': sentiment,
            'positive': round(scores['pos'] * 100, 2),
            'negative': round(scores['neg'] * 100, 2),
            'neutral': round(scores['neu'] * 100, 2),
            'compound': round(scores['compound'], 3)
        }
        
    except Exception as e:
        app.logger.error(f"Sentiment analysis error: {str(e)}")
        raise ValueError(f"Sentiment analysis failed: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
@limiter.limit("5 per minute")
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        if not email or '@' not in email:
            return jsonify({'error': 'Invalid email address'}), 400
        if not password or len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
            
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400
            
        hashed_password = generate_password_hash(password)
        new_user = User(
            username=username,
            email=email,
            password=hashed_password
        )
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({'message': 'Registration successful'})
        
    except Exception as e:
        app.logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed. Please try again.'}), 500

@app.route('/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    try:
        data = request.get_json()
        identifier = data.get('identifier')
        password = data.get('password')
        
        if not identifier or not password:
            return jsonify({'error': 'Username/email and password are required'}), 400
            
        user = User.query.filter((User.username == identifier) | (User.email == identifier)).first()
        if not user or not check_password_hash(user.password, password):
            return jsonify({'error': 'Invalid credentials'}), 401
            
        session['user_id'] = user.id
        session.permanent = True
        
        return jsonify({
            'message': 'Login successful',
            'username': user.username,
            'preferences': user.preferences
        })
        
    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed. Please try again.'}), 500

@app.route('/logout', methods=['POST'])
def logout():
    try:
        session.clear()
        return jsonify({'message': 'Logged out successfully'})
    except Exception as e:
        app.logger.error(f"Logout error: {str(e)}")
        return jsonify({'error': 'Logout failed'}), 500

@app.route('/profile', methods=['GET'])
def profile():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
            
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        return jsonify({
            'username': user.username,
            'preferences': user.preferences
        })
        
    except Exception as e:
        app.logger.error(f"Profile error: {str(e)}")
        return jsonify({'error': 'Failed to fetch profile'}), 500

@app.route('/preferences', methods=['POST'])
def update_preferences():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
            
        data = request.get_json()
        summary_length = data.get('summary_length')
        
        if not summary_length or summary_length not in ['100', '150', '200']:
            return jsonify({'error': 'Invalid preference value'}), 400
            
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        user.preferences = summary_length
        db.session.commit()
        
        return jsonify({'message': 'Preferences updated successfully'})
        
    except Exception as e:
        app.logger.error(f"Preferences error: {str(e)}")
        return jsonify({'error': 'Failed to update preferences'}), 500

@app.route('/summarize', methods=['POST'])
@limiter.limit("10 per minute")
async def summarize():
    try:
        if 'user_id' not in session and 'trial_used' in session:
            return jsonify({'error': 'Please register to continue using the service'}), 401
        
        max_length = 150
        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            if user and user.preferences:
                try:
                    max_length = int(user.preferences)
                except ValueError:
                    pass
        
        text = ""
        if 'pdf_file' in request.files and request.files['pdf_file']:
            try:
                text = extract_text_from_pdf(request.files['pdf_file'])
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        else:
            text_input = request.form.get('text', '').strip()
            if text_input:
                if urlparse(text_input).scheme in ["http", "https"]:
                    try:
                        text = await fetch_url_content(text_input)
                    except Exception as e:
                        return jsonify({'error': str(e)}), 400
                else:
                    text = text_input
        
        if not text or len(text.strip()) < 20:
            return jsonify({'error': 'No valid content to summarize (minimum 20 characters required)'}), 400
        
        try:
            summary = summarize_text(text, max_length)
            app.logger.info(f"Generated summary (length: {len(summary)})")
            
            if 'user_id' not in session:
                session['trial_used'] = True
            
            return jsonify({
                'summary': summary
            })
            
        except Exception as e:
            app.logger.error(f"Summarization failed: {str(e)}")
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        app.logger.error(f"Unexpected error in /summarize: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze():
    try:
        if 'user_id' not in session and 'trial_used' in session:
            return jsonify({'error': 'Please register to continue using the service'}), 401
            
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text or len(text) < 10:
            return jsonify({'error': 'Text too short for analysis (minimum 10 characters required)'}), 400
            
        analysis = analyze_text(text)
        
        if 'user_id' not in session:
            session['trial_used'] = True
            
        return jsonify({
            'analysis': analysis
        })
        
    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/extract', methods=['POST'])
@limiter.limit("10 per minute")
async def extract():
    try:
        if 'user_id' not in session and 'trial_used' in session:
            return jsonify({'error': 'Please register to continue using the service'}), 401
            
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url or not urlparse(url).scheme:
            return jsonify({'error': 'Invalid URL provided'}), 400
            
        content = await fetch_url_content(url)
        
        if 'user_id' not in session:
            session['trial_used'] = True
            
        return jsonify({
            'content': content
        })
        
    except Exception as e:
        app.logger.error(f"Extraction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/keywords', methods=['POST'])
@limiter.limit("10 per minute")
def keywords():
    try:
        if 'user_id' not in session and 'trial_used' in session:
            return jsonify({'error': 'Please register to continue using the service'}), 401
            
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text or len(text) < 10:
            return jsonify({'error': 'Text too short for keyword extraction (minimum 10 characters required)'}), 400
            
        keywords = extract_keywords(text)
        
        if 'user_id' not in session:
            session['trial_used'] = True
            
        return jsonify({
            'keywords': keywords
        })
        
    except Exception as e:
        app.logger.error(f"Keyword extraction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/sentiment', methods=['POST'])
@limiter.limit("10 per minute")
def sentiment():
    try:
        if 'user_id' not in session and 'trial_used' in session:
            return jsonify({'error': 'Please register to continue using the service'}), 401
            
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text or len(text) < 10:
            return jsonify({'error': 'Text too short for sentiment analysis (minimum 10 characters required)'}), 400
            
        sentiment = analyze_sentiment(text)
        
        if 'user_id' not in session:
            session['trial_used'] = True
         
        return jsonify({
            'sentiment': sentiment
        })
        
    except Exception as e:
        app.logger.error(f"Sentiment analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    return "Application is running"

if __name__ == '__main__':
    app.run(debug=True)
