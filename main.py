# from flask import Flask, request, jsonify, session, render_template
# from flask_sqlalchemy import SQLAlchemy
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# from bs4 import BeautifulSoup
# from urllib.parse import urlparse
# import PyPDF2
# from io import BytesIO
# from dotenv import load_dotenv
# import os
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import re
# import asyncio
# import aiohttp
# from functools import lru_cache
# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address
# import logging
# from logging.handlers import RotatingFileHandler
# import secrets
# from datetime import timedelta, datetime, UTC
# import bleach
# import pymysql
# from threading import Lock
# from waitress import serve
# import pdf2image
# import pytesseract

# # Initialize NLTK resources
# nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('vader_lexicon', quiet=True)

# load_dotenv()
# pymysql.install_as_MySQLdb()

# app = Flask(__name__)

# # Configuration
# app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///summarizer.db')
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
# app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
# app.config['SESSION_COOKIE_SECURE'] = True
# app.config['SESSION_COOKIE_HTTPONLY'] = True
# app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# # Initialize SQLAlchemy
# db = SQLAlchemy(app)

# # Configure logging
# handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# app.logger.addHandler(handler)

# # RateLimit Model
# class RateLimit(db.Model):
#     __tablename__ = 'rate_limits'
#     id = db.Column(db.Integer, primary_key=True)
#     key = db.Column(db.String(255), nullable=False, unique=True)
#     expiry = db.Column(db.DateTime, nullable=False)
#     request_count = db.Column(db.Integer, nullable=False, default=0)

# # User Model
# class User(db.Model):
#     __tablename__ = 'users'
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password = db.Column(db.String(255), nullable=False)
#     preferences = db.Column(db.String(50), default='150')
#     is_active = db.Column(db.Boolean, default=True)
#     created_at = db.Column(db.DateTime, server_default=db.func.now())

# # Initialize database
# with app.app_context():
#     db.create_all()

# # Initialize T5 model
# model_lock = Lock()
# try:
#     tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")
# except Exception as e:
#     app.logger.error(f"Failed to initialize T5 model: {str(e)}")
#     tokenizer = None
#     model = None

# # Text processing utilities
# sid = SentimentIntensityAnalyzer()
# stop_words = set(stopwords.words('english'))

# limiter = Limiter(
#     app=app,
#     key_func=get_remote_address,
#     storage_uri="memory://",
#     strategy="fixed-window",
#     default_limits=["200 per day", "50 per hour"]
# )

# def check_rate_limit(key, limit, period):
#     now = datetime.now(UTC)
#     with db.session.begin():
#         rate_limit = db.session.query(RateLimit).filter_by(key=key).first()
#         if not rate_limit:
#             rate_limit = RateLimit(
#                 key=key,
#                 expiry=now + timedelta(seconds=period),
#                 request_count=1
#             )
#             db.session.add(rate_limit)
#         else:
#             expiry_aware = rate_limit.expiry.replace(tzinfo=UTC) if rate_limit.expiry.tzinfo is None else rate_limit.expiry
#             if expiry_aware < now:
#                 rate_limit.expiry = now + timedelta(seconds=period)
#                 rate_limit.request_count = 1
#             else:
#                 if rate_limit.request_count >= limit:
#                     return False
#                 rate_limit.request_count += 1
#         db.session.commit()
#     return True

# @lru_cache(maxsize=128)
# def summarize_text(text, max_length=150):
#     if not tokenizer or not model:
#         raise ValueError("Summarization model not available")
#     try:
#         if not text or len(text.strip()) < 20:
#             raise ValueError("Text too short for summarization (minimum 20 characters required)")
#         sanitized_text = bleach.clean(text[:20000])
#         input_text = "summarize: " + sanitized_text
#         input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
#         with model_lock:
#             summary_ids = model.generate(
#                 input_ids,
#                 max_length=max_length,
#                 min_length=min(30, max_length//2),
#                 length_penalty=2.0,
#                 num_beams=4,
#                 early_stopping=True
#             )
#             summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#             if not summary or len(summary.strip()) == 0:
#                 raise ValueError("Failed to generate meaningful summary")
#             return summary.strip()
#     except Exception as e:
#         app.logger.error(f"Summarization error: {str(e)}")
#         raise ValueError(f"Summarization failed: {str(e)}")

# async def fetch_url_content(url):
#     try:
#         if not url or not urlparse(url).scheme:
#             raise ValueError("Invalid URL provided")
#         headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
#         timeout = aiohttp.ClientTimeout(total=15)
#         async with aiohttp.ClientSession(headers=headers) as session:
#             async with session.get(url, timeout=timeout) as response:
#                 if response.status != 200:
#                     raise ValueError(f"HTTP Error {response.status}")
#                 content = await response.text()
#                 soup = BeautifulSoup(content, "html.parser")
#                 for element in soup(["script", "style", "iframe", "noscript", "header", "footer", "nav"]):
#                     element.decompose()
#                 text = ' '.join([el.get_text(separator=" ", strip=True) for el in soup.find_all(['p', 'h1', 'h2', 'h3'])])
#                 if not text.strip():
#                     raise ValueError("No readable content found on page")
#                 return bleach.clean(text[:20000])
#     except Exception as e:
#         app.logger.error(f"URL fetch error: {str(e)}")
#         raise ValueError(f"Failed to fetch URL content: {str(e)}")

# def analyze_text(text):
#     try:
#         if not text or len(text.strip()) < 10:
#             raise ValueError("Text too short for analysis (minimum 10 characters required)")
#         clean_text = re.sub(r'[^\w\s]', '', text.lower())
#         words = word_tokenize(clean_text)
#         if not words:
#             raise ValueError("No valid words found in text")
#         word_count = len(words)
#         unique_words = len(set(words))
#         sentences = sent_tokenize(text)
#         sentence_count = len(sentences) if sentences else 1
#         avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
#         avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
#         readability = "Basic" if avg_sentence_length < 15 else "Intermediate" if avg_sentence_length < 25 else "Advanced"
#         return {
#             'word_count': word_count,
#             'unique_words': unique_words,
#             'sentence_count': sentence_count,
#             'avg_word_length': round(avg_word_length, 2),
#             'avg_sentence_length': round(avg_sentence_length, 2),
#             'readability': readability
#         }
#     except Exception as e:
#         app.logger.error(f"Text analysis error: {str(e)}")
#         raise ValueError(f"Text analysis failed: {str(e)}")

# def extract_keywords(text, top_n=10):
#     try:
#         if not text or len(text.strip()) < 10:
#             raise ValueError("Text too short for keyword extraction")
#         clean_text = re.sub(r'[^\w\s]', '', text.lower())
#         words = word_tokenize(clean_text)
#         words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
#         freq_dist = nltk.FreqDist(words)
#         return [word for word, _ in freq_dist.most_common(top_n)]
#     except Exception as e:
#         app.logger.error(f"Keyword extraction error: {str(e)}")
#         raise ValueError(f"Keyword extraction failed: {str(e)}")

# def analyze_sentiment(text):
#     try:
#         if not text or len(text.strip()) < 10:
#             raise ValueError("Text too short for sentiment analysis")
#         scores = sid.polarity_scores(text)
#         sentiment = "Positive" if scores['compound'] > 0.05 else "Negative" if scores['compound'] < -0.05 else "Neutral"
#         return {
#             'sentiment': sentiment,
#             'positive': round(scores['pos'] * 100, 2),
#             'negative': round(scores['neg'] * 100, 2),
#             'neutral': round(scores['neu'] * 100, 2),
#             'compound': round(scores['compound'], 3)
#         }
#     except Exception as e:
#         app.logger.error(f"Sentiment analysis error: {str(e)}")
#         raise ValueError(f"Sentiment analysis failed: {str(e)}")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/register', methods=['POST'])
# @limiter.limit("5 per minute")
# def register():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"register:{ip_address}", 5, 60):
#             return jsonify({'error': 'Rate limit exceeded: 5 per minute'}), 429
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
#         username, email, password = data.get('username'), data.get('email'), data.get('password')
#         if not username or len(username) < 3:
#             return jsonify({'error': 'Username must be at least 3 characters'}), 400
#         if not email or '@' not in email:
#             return jsonify({'error': 'Invalid email address'}), 400
#         if not password or len(password) < 5:
#             return jsonify({'error': 'Password must be at least 5 characters'}), 400
#         if User.query.filter_by(username=username).first():
#             return jsonify({'error': 'Username already exists'}), 400
#         if User.query.filter_by(email=email).first():
#             return jsonify({'error': 'Email already registered'}), 400
#         new_user = User(username=username, email=email, password=password)
#         db.session.add(new_user)
#         db.session.commit()
#         session['user_id'] = new_user.id
#         session.permanent = True
#         app.logger.info(f"User {username} registered")
#         return jsonify({'message': 'Registration successful', 'username': new_user.username, 'preferences': new_user.preferences}), 201
#     except Exception as e:
#         db.session.rollback()
#         app.logger.error(f"Registration error: {str(e)}")
#         return jsonify({'error': 'Registration failed'}), 500

# @app.route('/login', methods=['POST'])
# @limiter.limit("10 per minute")
# def login():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"login:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
#         identifier, password = data.get('identifier'), data.get('password')
#         if not identifier or not password:
#             return jsonify({'error': 'Username/email and password required'}), 400
#         user = User.query.filter((User.username == identifier) | (User.email == identifier)).first()
#         if not user or user.password != password:
#             return jsonify({'error': 'Invalid credentials'}), 401
#         session['user_id'] = user.id
#         session.permanent = True
#         app.logger.info(f"User {user.username} logged in")
#         return jsonify({'message': 'Login successful', 'username': user.username, 'preferences': user.preferences})
#     except Exception as e:
#         app.logger.error(f"Login error: {str(e)}")
#         return jsonify({'error': 'Login failed'}), 500

# @app.route('/logout', methods=['POST'])
# @limiter.limit("10 per minute")
# def logout():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"logout:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
#         if 'user_id' not in session:
#             return jsonify({'error': 'Not authenticated'}), 401
#         user_id = session.get('user_id')
#         session.clear()
#         app.logger.info(f"User ID {user_id} logged out")
#         return jsonify({'success': True})
#     except Exception as e:
#         app.logger.error(f"Logout error: {str(e)}")
#         return jsonify({'error': 'Logout failed'}), 500

# @app.route('/profile', methods=['GET'])
# def profile():
#     try:
#         if 'user_id' not in session:
#             return jsonify({'error': 'Not authenticated'}), 401
#         user = db.session.get(User, session['user_id'])
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
#         return jsonify({
#             'username': user.username,
#             'preferences': user.preferences,
#             'joinDate': user.created_at.strftime('%B %d, %Y') if user.created_at else 'Unknown'
#         })
#     except Exception as e:
#         app.logger.error(f"Profile error: {str(e)}")
#         return jsonify({'error': 'Failed to fetch profile'}), 500

# @app.route('/preferences', methods=['POST'])
# def update_preferences():
#     try:
#         if 'user_id' not in session:
#             return jsonify({'error': 'Not authenticated'}), 401
#         data = request.get_json()
#         if not data or 'summary_length' not in data:
#             return jsonify({'error': 'No preference data provided'}), 400
#         summary_length = data['summary_length']
#         if summary_length not in ['100', '150', '200']:
#             return jsonify({'error': 'Invalid preference value'}), 400
#         user = db.session.get(User, session['user_id'])
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
#         user.preferences = summary_length
#         db.session.commit()
#         app.logger.info(f"User {user.username} updated preferences")
#         return jsonify({'message': 'Preferences updated'})
#     except Exception as e:
#         db.session.rollback()
#         app.logger.error(f"Preferences error: {str(e)}")
#         return jsonify({'error': 'Failed to update preferences'}), 500

# @app.route('/summarize', methods=['POST'])
# @limiter.limit("10 per minute")
# def summarize():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"summarize:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
            
#         if 'user_id' not in session and 'trial_used' in session:
#             return jsonify({'error': 'Please register to continue'}), 401
            
#         max_length = 150
#         if 'user_id' in session:
#             user = db.session.get(User, session['user_id'])
#             if user and user.preferences:
#                 max_length = int(user.preferences)
                
#         data = request.get_json()
#         if not data or 'text' not in data:
#             return jsonify({'error': 'No text provided. Please enter text to summarize.'}), 400
            
#         text = data['text'].strip()
#         if not text:
#             return jsonify({'error': 'Text is empty. Please enter text to summarize.'}), 400
            
#         if len(text.strip()) < 20:
#             return jsonify({'error': 'Text too short. Please provide content with at least 20 characters.'}), 400
            
#         summary = summarize_text(text, max_length)
        
#         if 'user_id' not in session:
#             session['trial_used'] = True
            
#         return jsonify({'summary': summary})
        
#     except ValueError as ve:
#         app.logger.error(f"Summarization failed: {str(ve)}")
#         return jsonify({'error': str(ve)}), 400
#     except Exception as e:
#         app.logger.error(f"Summarization failed unexpectedly: {str(e)}")
#         return jsonify({'error': 'An unexpected error occurred during summarization. Please try again.'}), 500

# @app.route('/analyze', methods=['POST'])
# @limiter.limit("10 per minute")
# def analyze():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"analyze:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
#         if 'user_id' not in session and 'trial_used' in session:
#             return jsonify({'error': 'Please register to continue'}), 401
#         data = request.get_json()
#         if not data or 'text' not in data:
#             return jsonify({'error': 'No text provided'}), 400
#         text = data['text'].strip()
#         if not text:
#             return jsonify({'error': 'Text is empty. Please provide text to analyze.'}), 400
#         if len(text) < 10:
#             return jsonify({'error': 'Text too short for analysis (minimum 10 characters required)'}), 400
#         analysis = analyze_text(text)
#         if 'user_id' not in session:
#             session['trial_used'] = True
#         return jsonify({'analysis': analysis})
#     except ValueError as ve:
#         app.logger.error(f"Analyze error: {str(ve)}")
#         return jsonify({'error': str(ve)}), 400
#     except Exception as e:
#         app.logger.error(f"Analyze error: {str(e)}")
#         return jsonify({'error': 'An unexpected error occurred during analysis'}), 500

# @app.route('/extract', methods=['POST'])
# @limiter.limit("10 per minute")
# async def extract():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"extract:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
#         if 'user_id' not in session and 'trial_used' in session:
#             return jsonify({'error': 'Please register to continue'}), 401
#         data = request.get_json()
#         if not data or 'url' not in data:
#             return jsonify({'error': 'No URL provided'}), 400
#         url = data['url'].strip()
#         if not urlparse(url).scheme:
#             return jsonify({'error': 'Invalid URL'}), 400
#         content = await fetch_url_content(url)
#         if 'user_id' not in session:
#             session['trial_used'] = True
#         return jsonify({'content': content})
#     except Exception as e:
#         app.logger.error(f"Extract error: {str(e)}")
#         return jsonify({'error': str(e)}), 400

# @app.route('/keywords', methods=['POST'])
# @limiter.limit("10 per minute")
# def keywords():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"keywords:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
#         if 'user_id' not in session and 'trial_used' in session:
#             return jsonify({'error': 'Please register to continue'}), 401
#         data = request.get_json()
#         if not data or 'text' not in data:
#             return jsonify({'error': 'No text provided'}), 400
#         text = data['text'].strip()
#         if len(text) < 10:
#             return jsonify({'error': 'Text too short for keyword extraction'}), 400
#         keywords = extract_keywords(text)
#         if 'user_id' not in session:
#             session['trial_used'] = True
#         return jsonify({'keywords': keywords})
#     except Exception as e:
#         app.logger.error(f"Keywords error: {str(e)}")
#         return jsonify({'error': str(e)}), 400

# @app.route('/sentiment', methods=['POST'])
# @limiter.limit("10 per minute")
# def sentiment():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"sentiment:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
#         if 'user_id' not in session and 'trial_used' in session:
#             return jsonify({'error': 'Please register to continue'}), 401
#         data = request.get_json()
#         if not data or 'text' not in data:
#             return jsonify({'error': 'No text provided'}), 400
#         text = data['text'].strip()
#         if len(text) < 10:
#             return jsonify({'error': 'Text too short for sentiment analysis'}), 400
#         sentiment = analyze_sentiment(text)
#         if 'user_id' not in session:
#             session['trial_used'] = True
#         return jsonify({'sentiment': sentiment})
#     except Exception as e:
#         app.logger.error(f"Sentiment error: {str(e)}")
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     serve(app, host='0.0.0.0', port=5000)


# from flask import Flask, request, jsonify, session, render_template
# from flask_sqlalchemy import SQLAlchemy
# from bs4 import BeautifulSoup
# from urllib.parse import urlparse
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import re
# import asyncio
# import aiohttp
# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address
# import logging
# from logging.handlers import RotatingFileHandler
# import secrets
# from datetime import timedelta, datetime, UTC
# import bleach
# import pymysql
# from waitress import serve

# # Initialize NLTK resources
# nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('vader_lexicon', quiet=True)

# pymysql.install_as_MySQLdb()

# app = Flask(__name__)

# # Configuration
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///summarizer.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['SECRET_KEY'] = secrets.token_hex(32)
# app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
# app.config['SESSION_COOKIE_SECURE'] = False  # For local testing
# app.config['SESSION_COOKIE_HTTPONLY'] = True
# app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# # Initialize SQLAlchemy
# db = SQLAlchemy(app)

# # Configure logging
# handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# app.logger.addHandler(handler)
# app.logger.setLevel(logging.DEBUG)

# # RateLimit Model
# class RateLimit(db.Model):
#     __tablename__ = 'rate_limits'
#     id = db.Column(db.Integer, primary_key=True)
#     key = db.Column(db.String(255), nullable=False, unique=True)
#     expiry = db.Column(db.DateTime, nullable=False)
#     request_count = db.Column(db.Integer, nullable=False, default=0)

# # User Model
# class User(db.Model):
#     __tablename__ = 'users'
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password = db.Column(db.String(255), nullable=False)
#     preferences = db.Column(db.String(50), default='150')
#     is_active = db.Column(db.Boolean, default=True)
#     created_at = db.Column(db.DateTime, server_default=db.func.now())

# # Initialize database
# with app.app_context():
#     db.create_all()

# # Text processing utilities
# sid = SentimentIntensityAnalyzer()
# stop_words = set(stopwords.words('english'))

# limiter = Limiter(
#     app=app,
#     key_func=get_remote_address,
#     storage_uri="memory://",
#     strategy="fixed-window",
#     default_limits=["200 per day", "50 per hour"]
# )

# def check_rate_limit(key, limit, period):
#     now = datetime.now(UTC)
#     with db.session.begin():
#         rate_limit = db.session.query(RateLimit).filter_by(key=key).first()
#         if not rate_limit:
#             rate_limit = RateLimit(
#                 key=key,
#                 expiry=now + timedelta(seconds=period),
#                 request_count=1
#             )
#             db.session.add(rate_limit)
#         else:
#             expiry_aware = rate_limit.expiry.replace(tzinfo=UTC) if rate_limit.expiry.tzinfo is None else rate_limit.expiry
#             if expiry_aware < now:
#                 rate_limit.expiry = now + timedelta(seconds=period)
#                 rate_limit.request_count = 1
#             else:
#                 if rate_limit.request_count >= limit:
#                     return False
#                 rate_limit.request_count += 1
#         db.session.commit()
#     return True

# def summarize_text(text, max_length=150):
#     """Simple summarization using NLTK"""
#     try:
#         if not text or len(text.strip()) < 20:
#             raise ValueError("Text too short for summarization (minimum 20 characters required)")
#         sanitized_text = bleach.clean(text[:10000])
#         sentences = sent_tokenize(sanitized_text)
#         if not sentences:
#             raise ValueError("No sentences detected in text")
#         # Take the first few sentences up to max_length words
#         summary_words = []
#         word_count = 0
#         for sentence in sentences:
#             words = word_tokenize(sentence)
#             if word_count + len(words) <= max_length:
#                 summary_words.extend(words)
#                 word_count += len(words)
#             else:
#                 break
#         summary = ' '.join(summary_words) or sentences[0][:max_length]
#         return summary.strip()
#     except Exception as e:
#         app.logger.error(f"Summarization error: {str(e)}")
#         raise ValueError(f"Summarization failed: {str(e)}")

# async def fetch_url_content(url):
#     try:
#         if not url or not urlparse(url).scheme:
#             raise ValueError("Invalid URL provided")
#         headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
#         timeout = aiohttp.ClientTimeout(total=15)
#         async with aiohttp.ClientSession(headers=headers) as session:
#             async with session.get(url, timeout=timeout) as response:
#                 if response.status != 200:
#                     raise ValueError(f"HTTP Error {response.status}")
#                 content = await response.text()
#                 soup = BeautifulSoup(content, "html.parser")
#                 for element in soup(["script", "style", "iframe", "noscript", "header", "footer", "nav"]):
#                     element.decompose()
#                 text = ' '.join([el.get_text(separator=" ", strip=True) for el in soup.find_all(['p', 'h1', 'h2', 'h3'])])
#                 if not text.strip():
#                     raise ValueError("No readable content found on page")
#                 return bleach.clean(text[:10000])
#     except Exception as e:
#         app.logger.error(f"URL fetch error: {str(e)}")
#         raise ValueError(f"Failed to fetch URL content: {str(e)}")

# def analyze_text(text):
#     try:
#         if not text or len(text.strip()) < 10:
#             raise ValueError("Text too short for analysis (minimum 10 characters required)")
#         clean_text = re.sub(r'[^\w\s]', '', text.lower())
#         words = word_tokenize(clean_text)
#         if not words:
#             raise ValueError("No valid words found in text")
#         word_count = len(words)
#         unique_words = len(set(words))
#         sentences = sent_tokenize(text)
#         sentence_count = len(sentences) if sentences else 1
#         avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
#         avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
#         readability = "Basic" if avg_sentence_length < 15 else "Intermediate" if avg_sentence_length < 25 else "Advanced"
#         return {
#             'word_count': word_count,
#             'unique_words': unique_words,
#             'sentence_count': sentence_count,
#             'avg_word_length': round(avg_word_length, 2),
#             'avg_sentence_length': round(avg_sentence_length, 2),
#             'readability': readability
#         }
#     except Exception as e:
#         app.logger.error(f"Text analysis error: {str(e)}")
#         raise ValueError(f"Text analysis failed: {str(e)}")

# def extract_keywords(text, top_n=10):
#     try:
#         if not text or len(text.strip()) < 10:
#             raise ValueError("Text too short for keyword extraction")
#         clean_text = re.sub(r'[^\w\s]', '', text.lower())
#         words = word_tokenize(clean_text)
#         words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
#         freq_dist = nltk.FreqDist(words)
#         return [word for word, _ in freq_dist.most_common(top_n)]
#     except Exception as e:
#         app.logger.error(f"Keyword extraction error: {str(e)}")
#         raise ValueError(f"Keyword extraction failed: {str(e)}")

# def analyze_sentiment(text):
#     try:
#         if not text or len(text.strip()) < 10:
#             raise ValueError("Text too short for sentiment analysis")
#         scores = sid.polarity_scores(text)
#         sentiment = "Positive" if scores['compound'] > 0.05 else "Negative" if scores['compound'] < -0.05 else "Neutral"
#         return {
#             'sentiment': sentiment,
#             'positive': round(scores['pos'] * 100, 2),
#             'negative': round(scores['neg'] * 100, 2),
#             'neutral': round(scores['neu'] * 100, 2),
#             'compound': round(scores['compound'], 3)
#         }
#     except Exception as e:
#         app.logger.error(f"Sentiment analysis error: {str(e)}")
#         raise ValueError(f"Sentiment analysis failed: {str(e)}")

# @app.route('/')
# def home():
#     app.logger.debug("Serving home page")
#     return render_template('index.html')

# @app.route('/register', methods=['POST'])
# @limiter.limit("5 per minute")
# def register():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"register:{ip_address}", 5, 60):
#             return jsonify({'error': 'Rate limit exceeded: 5 per minute'}), 429
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
#         username, email, password = data.get('username'), data.get('email'), data.get('password')
#         if not username or len(username) < 3:
#             return jsonify({'error': 'Username must be at least 3 characters'}), 400
#         if not email or '@' not in email:
#             return jsonify({'error': 'Invalid email address'}), 400
#         if not password or len(password) < 5:
#             return jsonify({'error': 'Password must be at least 5 characters'}), 400
#         if User.query.filter_by(username=username).first():
#             return jsonify({'error': 'Username already exists'}), 400
#         if User.query.filter_by(email=email).first():
#             return jsonify({'error': 'Email already registered'}), 400
#         new_user = User(username=username, email=email, password=password)
#         db.session.add(new_user)
#         db.session.commit()
#         session['user_id'] = new_user.id
#         session.permanent = True
#         app.logger.info(f"User {username} registered")
#         return jsonify({'message': 'Registration successful', 'username': new_user.username, 'preferences': new_user.preferences}), 201
#     except Exception as e:
#         db.session.rollback()
#         app.logger.error(f"Registration error: {str(e)}")
#         return jsonify({'error': 'Registration failed'}), 500

# @app.route('/login', methods=['POST'])
# @limiter.limit("10 per minute")
# def login():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"login:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
#         identifier, password = data.get('identifier'), data.get('password')
#         if not identifier or not password:
#             return jsonify({'error': 'Username/email and password required'}), 400
#         user = User.query.filter((User.username == identifier) | (User.email == identifier)).first()
#         if not user or user.password != password:
#             return jsonify({'error': 'Invalid credentials'}), 401
#         session['user_id'] = user.id
#         session.permanent = True
#         app.logger.info(f"User {user.username} logged in")
#         return jsonify({'message': 'Login successful', 'username': user.username, 'preferences': user.preferences})
#     except Exception as e:
#         app.logger.error(f"Login error: {str(e)}")
#         return jsonify({'error': 'Login failed'}), 500

# @app.route('/logout', methods=['POST'])
# @limiter.limit("10 per minute")
# def logout():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"logout:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
#         if 'user_id' not in session:
#             return jsonify({'error': 'Not authenticated'}), 401
#         user_id = session.get('user_id')
#         session.clear()
#         app.logger.info(f"User ID {user_id} logged out")
#         return jsonify({'success': True})
#     except Exception as e:
#         app.logger.error(f"Logout error: {str(e)}")
#         return jsonify({'error': 'Logout failed'}), 500

# @app.route('/profile', methods=['GET'])
# def profile():
#     try:
#         if 'user_id' not in session:
#             return jsonify({'error': 'Not authenticated'}), 401
#         user = db.session.get(User, session['user_id'])
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
#         return jsonify({
#             'username': user.username,
#             'preferences': user.preferences,
#             'joinDate': user.created_at.strftime('%B %d, %Y') if user.created_at else 'Unknown'
#         })
#     except Exception as e:
#         app.logger.error(f"Profile error: {str(e)}")
#         return jsonify({'error': 'Failed to fetch profile'}), 500

# @app.route('/preferences', methods=['POST'])
# def update_preferences():
#     try:
#         if 'user_id' not in session:
#             return jsonify({'error': 'Not authenticated'}), 401
#         data = request.get_json()
#         if not data or 'summary_length' not in data:
#             return jsonify({'error': 'No preference data provided'}), 400
#         summary_length = data['summary_length']
#         if summary_length not in ['100', '150', '200']:
#             return jsonify({'error': 'Invalid preference value'}), 400
#         user = db.session.get(User, session['user_id'])
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
#         user.preferences = summary_length
#         db.session.commit()
#         app.logger.info(f"User {user.username} updated preferences")
#         return jsonify({'message': 'Preferences updated'})
#     except Exception as e:
#         db.session.rollback()
#         app.logger.error(f"Preferences error: {str(e)}")
#         return jsonify({'error': 'Failed to update preferences'}), 500

# @app.route('/summarize', methods=['POST'])
# @limiter.limit("10 per minute")
# def summarize():
#     try:
#         ip_address = get_remote_address()
#         app.logger.debug(f"Processing summarize request from {ip_address}")
#         if not check_rate_limit(f"summarize:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
            
#         if 'user_id' not in session and 'trial_used' in session:
#             return jsonify({'error': 'Please register to continue'}), 401
            
#         max_length = 150
#         if 'user_id' in session:
#             user = db.session.get(User, session['user_id'])
#             if user and user.preferences:
#                 max_length = int(user.preferences)
                
#         data = request.get_json()
#         if not data or 'text' not in data:
#             return jsonify({'error': 'No text provided. Please enter text to summarize.'}), 400
            
#         text = data['text'].strip()
#         if not text:
#             return jsonify({'error': 'Text is empty. Please enter text to summarize.'}), 400
            
#         if len(text.strip()) < 20:
#             return jsonify({'error': 'Text too short. Please provide content with at least 20 characters.'}), 400
            
#         summary = summarize_text(text, max_length)
        
#         if 'user_id' not in session:
#             session['trial_used'] = True
            
#         app.logger.debug(f"Summary generated: {summary[:50]}...")
#         return jsonify({'summary': summary})
        
#     except ValueError as ve:
#         app.logger.error(f"Summarization failed: {str(ve)}")
#         return jsonify({'error': str(ve)}), 400
#     except Exception as e:
#         app.logger.error(f"Summarization failed unexpectedly: {str(e)}", exc_info=True)
#         return jsonify({'error': 'An unexpected error occurred during summarization. Please try again.'}), 500

# @app.route('/analyze', methods=['POST'])
# @limiter.limit("10 per minute")
# def analyze():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"analyze:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
#         if 'user_id' not in session and 'trial_used' in session:
#             return jsonify({'error': 'Please register to continue'}), 401
#         data = request.get_json()
#         if not data or 'text' not in data:
#             return jsonify({'error': 'No text provided'}), 400
#         text = data['text'].strip()
#         if not text:
#             return jsonify({'error': 'Text is empty. Please provide text to analyze.'}), 400
#         if len(text) < 10:
#             return jsonify({'error': 'Text too short for analysis (minimum 10 characters required)'}), 400
#         analysis = analyze_text(text)
#         if 'user_id' not in session:
#             session['trial_used'] = True
#         return jsonify({'analysis': analysis})
#     except ValueError as ve:
#         app.logger.error(f"Analyze error: {str(ve)}")
#         return jsonify({'error': str(ve)}), 400
#     except Exception as e:
#         app.logger.error(f"Analyze error: {str(e)}")
#         return jsonify({'error': 'An unexpected error occurred during analysis'}), 500

# @app.route('/extract', methods=['POST'])
# @limiter.limit("10 per minute")
# async def extract():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"extract:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
#         if 'user_id' not in session and 'trial_used' in session:
#             return jsonify({'error': 'Please register to continue'}), 401
#         data = request.get_json()
#         if not data or 'url' not in data:
#             return jsonify({'error': 'No URL provided'}), 400
#         url = data['url'].strip()
#         if not urlparse(url).scheme:
#             return jsonify({'error': 'Invalid URL'}), 400
#         content = await fetch_url_content(url)
#         if 'user_id' not in session:
#             session['trial_used'] = True
#         return jsonify({'content': content})
#     except Exception as e:
#         app.logger.error(f"Extract error: {str(e)}")
#         return jsonify({'error': str(e)}), 400

# @app.route('/keywords', methods=['POST'])
# @limiter.limit("10 per minute")
# def keywords():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"keywords:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
#         if 'user_id' not in session and 'trial_used' in session:
#             return jsonify({'error': 'Please register to continue'}), 401
#         data = request.get_json()
#         if not data or 'text' not in data:
#             return jsonify({'error': 'No text provided'}), 400
#         text = data['text'].strip()
#         if len(text) < 10:
#             return jsonify({'error': 'Text too short for keyword extraction'}), 400
#         keywords = extract_keywords(text)
#         if 'user_id' not in session:
#             session['trial_used'] = True
#         return jsonify({'keywords': keywords})
#     except Exception as e:
#         app.logger.error(f"Keywords error: {str(e)}")
#         return jsonify({'error': str(e)}), 400

# @app.route('/sentiment', methods=['POST'])
# @limiter.limit("10 per minute")
# def sentiment():
#     try:
#         ip_address = get_remote_address()
#         if not check_rate_limit(f"sentiment:{ip_address}", 10, 60):
#             return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
#         if 'user_id' not in session and 'trial_used' in session:
#             return jsonify({'error': 'Please register to continue'}), 401
#         data = request.get_json()
#         if not data or 'text' not in data:
#             return jsonify({'error': 'No text provided'}), 400
#         text = data['text'].strip()
#         if len(text) < 10:
#             return jsonify({'error': 'Text too short for sentiment analysis'}), 400
#         sentiment = analyze_sentiment(text)
#         if 'user_id' not in session:
#             session['trial_used'] = True
#         return jsonify({'sentiment': sentiment})
#     except Exception as e:
#         app.logger.error(f"Sentiment error: {str(e)}")
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.logger.debug("Starting Waitress server...")
#     serve(app, host='0.0.0.0', port=5000, threads=4)


from flask import Flask, request, jsonify, session, render_template
from flask_sqlalchemy import SQLAlchemy
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import asyncio
import aiohttp
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from logging.handlers import RotatingFileHandler
import secrets
from datetime import timedelta, datetime, UTC
import bleach
import pymysql
from waitress import serve

# Initialize NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

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
db = SQLAlchemy(app)

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

# RateLimit Model
class RateLimit(db.Model):
    __tablename__ = 'rate_limits'
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(255), nullable=False, unique=True)
    expiry = db.Column(db.DateTime, nullable=False)
    request_count = db.Column(db.Integer, nullable=False, default=0)

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

# Text processing utilities
sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",
    strategy="fixed-window",
    default_limits=["200 per day", "50 per hour"]
)

def check_rate_limit(key, limit, period):
    now = datetime.now(UTC)
    with db.session.begin():
        rate_limit = db.session.query(RateLimit).filter_by(key=key).first()
        if not rate_limit:
            rate_limit = RateLimit(
                key=key,
                expiry=now + timedelta(seconds=period),
                request_count=1
            )
            db.session.add(rate_limit)
        else:
            expiry_aware = rate_limit.expiry.replace(tzinfo=UTC) if rate_limit.expiry.tzinfo is None else rate_limit.expiry
            if expiry_aware < now:
                rate_limit.expiry = now + timedelta(seconds=period)
                rate_limit.request_count = 1
            else:
                if rate_limit.request_count >= limit:
                    return False
                rate_limit.request_count += 1
        db.session.commit()
    return True

def summarize_text(text, max_length=150):
    """Simple summarization using NLTK"""
    try:
        if not text or len(text.strip()) < 20:
            raise ValueError("Text too short for summarization (minimum 20 characters required)")
        sanitized_text = bleach.clean(text[:10000])
        sentences = sent_tokenize(sanitized_text)
        if not sentences:
            raise ValueError("No sentences detected in text")
        # Take the first few sentences up to max_length words
        summary_words = []
        word_count = 0
        for sentence in sentences:
            words = word_tokenize(sentence)
            if word_count + len(words) <= max_length:
                summary_words.extend(words)
                word_count += len(words)
            else:
                break
        summary = ' '.join(summary_words) or sentences[0][:max_length]
        return summary.strip()
    except Exception as e:
        app.logger.error(f"Summarization error: {str(e)}")
        raise ValueError(f"Summarization failed: {str(e)}")

async def fetch_url_content(url):
    try:
        if not url or not urlparse(url).scheme:
            raise ValueError("Invalid URL provided")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP Error {response.status}")
                content = await response.text()
                soup = BeautifulSoup(content, "html.parser")
                for element in soup(["script", "style", "iframe", "noscript", "header", "footer", "nav"]):
                    element.decompose()
                text = ' '.join([el.get_text(separator=" ", strip=True) for el in soup.find_all(['p', 'h1', 'h2', 'h3'])])
                if not text.strip():
                    raise ValueError("No readable content found on page")
                return bleach.clean(text[:10000])
    except Exception as e:
        app.logger.error(f"URL fetch error: {str(e)}")
        raise ValueError(f"Failed to fetch URL content: {str(e)}")

def analyze_text(text):
    try:
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for analysis (minimum 10 characters required)")
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = word_tokenize(clean_text)
        if not words:
            raise ValueError("No valid words found in text")
        word_count = len(words)
        unique_words = len(set(words))
        sentences = sent_tokenize(text)
        sentence_count = len(sentences) if sentences else 1
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        readability = "Basic" if avg_sentence_length < 15 else "Intermediate" if avg_sentence_length < 25 else "Advanced"
        return {
            'word_count': word_count,
            'unique_words': unique_words,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'readability': readability
        }
    except Exception as e:
        app.logger.error(f"Text analysis error: {str(e)}")
        raise ValueError(f"Text analysis failed: {str(e)}")

def extract_keywords(text, top_n=10):
    try:
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for keyword extraction")
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = word_tokenize(clean_text)
        words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
        freq_dist = nltk.FreqDist(words)
        return [word for word, _ in freq_dist.most_common(top_n)]
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
    app.logger.debug("Serving home page")
    return render_template('index.html')

@app.route('/register', methods=['POST'])
@limiter.limit("5 per minute")
def register():
    try:
        ip_address = get_remote_address()
        if not check_rate_limit(f"register:{ip_address}", 5, 60):
            return jsonify({'error': 'Rate limit exceeded: 5 per minute'}), 429
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        username, email, password = data.get('username'), data.get('email'), data.get('password')
        if not username or len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        if not email or '@' not in email:
            return jsonify({'error': 'Invalid email address'}), 400
        if not password or len(password) < 5:
            return jsonify({'error': 'Password must be at least 5 characters'}), 400
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        session['user_id'] = new_user.id
        session.permanent = True
        app.logger.info(f"User {username} registered with user_id {new_user.id}")
        return jsonify({'message': 'Registration successful', 'username': new_user.username, 'preferences': new_user.preferences}), 201
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    try:
        ip_address = get_remote_address()
        if not check_rate_limit(f"login:{ip_address}", 10, 60):
            return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        identifier, password = data.get('identifier'), data.get('password')
        if not identifier or not password:
            return jsonify({'error': 'Username/email and password required'}), 400
        user = User.query.filter((User.username == identifier) | (User.email == identifier)).first()
        if not user or user.password != password:
            return jsonify({'error': 'Invalid credentials'}), 401
        session['user_id'] = user.id
        session.permanent = True
        app.logger.info(f"User {user.username} logged in with user_id {user.id}, session: {session.get('user_id')}")
        return jsonify({'message': 'Login successful', 'username': user.username, 'preferences': user.preferences})
    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/logout', methods=['POST'])
@limiter.limit("10 per minute")
def logout():
    try:
        ip_address = get_remote_address()
        if not check_rate_limit(f"logout:{ip_address}", 10, 60):
            return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        user_id = session.get('user_id')
        session.clear()
        app.logger.info(f"User ID {user_id} logged out")
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Logout error: {str(e)}")
        return jsonify({'error': 'Logout failed'}), 500

@app.route('/profile', methods=['GET'])
def profile():
    try:
        app.logger.debug(f"Profile request, session user_id: {session.get('user_id')}")
        if 'user_id' not in session:
            app.logger.warning("Profile access denied: No user_id in session")
            return jsonify({'error': 'Not authenticated'}), 401
        user = db.session.get(User, session['user_id'])
        if not user:
            app.logger.error(f"User not found for user_id: {session['user_id']}")
            return jsonify({'error': 'User not found'}), 404
        app.logger.info(f"Profile accessed for user: {user.username}")
        return jsonify({
            'username': user.username,
            'preferences': user.preferences,
            'joinDate': user.created_at.strftime('%B %d, %Y') if user.created_at else 'Unknown'
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
        if not data or 'summary_length' not in data:
            return jsonify({'error': 'No preference data provided'}), 400
        summary_length = data['summary_length']
        if summary_length not in ['100', '150', '200']:
            return jsonify({'error': 'Invalid preference value'}), 400
        user = db.session.get(User, session['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
        user.preferences = summary_length
        db.session.commit()
        app.logger.info(f"User {user.username} updated preferences")
        return jsonify({'message': 'Preferences updated'})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Preferences error: {str(e)}")
        return jsonify({'error': 'Failed to update preferences'}), 500

@app.route('/summarize', methods=['POST'])
@limiter.limit("10 per minute")
def summarize():
    try:
        ip_address = get_remote_address()
        app.logger.debug(f"Processing summarize request from {ip_address}")
        if not check_rate_limit(f"summarize:{ip_address}", 10, 60):
            return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
            
        if 'user_id' not in session and 'trial_used' in session:
            return jsonify({'error': 'Please register to continue'}), 401
            
        max_length = 150
        if 'user_id' in session:
            user = db.session.get(User, session['user_id'])
            if user and user.preferences:
                max_length = int(user.preferences)
                
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided. Please enter text to summarize.'}), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text is empty. Please enter text to summarize.'}), 400
            
        if len(text.strip()) < 20:
            return jsonify({'error': 'Text too short. Please provide content with at least 20 characters.'}), 400
            
        summary = summarize_text(text, max_length)
        
        if 'user_id' not in session:
            session['trial_used'] = True
            
        app.logger.debug(f"Summary generated: {summary[:50]}...")
        return jsonify({'summary': summary})
        
    except ValueError as ve:
        app.logger.error(f"Summarization failed: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Summarization failed unexpectedly: {str(e)}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during summarization. Please try again.'}), 500

@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze():
    try:
        ip_address = get_remote_address()
        if not check_rate_limit(f"analyze:{ip_address}", 10, 60):
            return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
        if 'user_id' not in session and 'trial_used' in session:
            return jsonify({'error': 'Please register to continue'}), 401
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text is empty. Please provide text to analyze.'}), 400
        if len(text) < 10:
            return jsonify({'error': 'Text too short for analysis (minimum 10 characters required)'}), 400
        analysis = analyze_text(text)
        if 'user_id' not in session:
            session['trial_used'] = True
        return jsonify({'analysis': analysis})
    except ValueError as ve:
        app.logger.error(f"Analyze error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Analyze error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred during analysis'}), 500

@app.route('/extract', methods=['POST'])
@limiter.limit("10 per minute")
async def extract():
    try:
        ip_address = get_remote_address()
        if not check_rate_limit(f"extract:{ip_address}", 10, 60):
            return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
        if 'user_id' not in session and 'trial_used' in session:
            return jsonify({'error': 'Please register to continue'}), 401
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400
        url = data['url'].strip()
        if not urlparse(url).scheme:
            return jsonify({'error': 'Invalid URL'}), 400
        content = await fetch_url_content(url)
        if 'user_id' not in session:
            session['trial_used'] = True
        return jsonify({'content': content})
    except Exception as e:
        app.logger.error(f"Extract error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/keywords', methods=['POST'])
@limiter.limit("10 per minute")
def keywords():
    try:
        ip_address = get_remote_address()
        if not check_rate_limit(f"keywords:{ip_address}", 10, 60):
            return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
        if 'user_id' not in session and 'trial_used' in session:
            return jsonify({'error': 'Please register to continue'}), 401
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        text = data['text'].strip()
        if len(text) < 10:
            return jsonify({'error': 'Text too short for keyword extraction'}), 400
        keywords = extract_keywords(text)
        if 'user_id' not in session:
            session['trial_used'] = True
        return jsonify({'keywords': keywords})
    except Exception as e:
        app.logger.error(f"Keywords error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/sentiment', methods=['POST'])
@limiter.limit("10 per minute")
def sentiment():
    try:
        ip_address = get_remote_address()
        if not check_rate_limit(f"sentiment:{ip_address}", 10, 60):
            return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
        if 'user_id' not in session and 'trial_used' in session:
            return jsonify({'error': 'Please register to continue'}), 401
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        text = data['text'].strip()
        if len(text) < 10:
            return jsonify({'error': 'Text too short for sentiment analysis'}), 400
        sentiment = analyze_sentiment(text)
        if 'user_id' not in session:
            session['trial_used'] = True
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        app.logger.error(f"Sentiment error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.logger.debug("Starting Waitress server...")
    serve(app, host='0.0.0.0', port=5000, threads=4)
