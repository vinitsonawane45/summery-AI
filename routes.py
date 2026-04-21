from flask import Blueprint, request, jsonify, session, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sqlalchemy.sql import func
from datetime import datetime, UTC, timedelta
from urllib.parse import urlparse
import logging
from bs4 import BeautifulSoup
import aiohttp
import bleach

from models import db, User, RateLimit
from agents import summarize_text, analyze_text, extract_keywords, analyze_sentiment

# Set up limiter. Note: this uses the main app's limiter if initialized properly,
# but since we need it in routes, we'll initialize a local limiter, or pass it.
# We will assume Limiter is imported and attached to app in app.py.

main_bp = Blueprint('main', __name__)

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
        logging.getLogger('app').error(f"URL fetch error: {str(e)}")
        raise ValueError(f"Failed to fetch URL content: {str(e)}")


@main_bp.route('/register', methods=['POST'])
def register():
    logger = logging.getLogger('app')
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
        if User.query.filter(func.lower(User.username) == username.lower()).first():
            return jsonify({'error': 'Username already exists'}), 400
        if User.query.filter(func.lower(User.email) == email.lower()).first():
            return jsonify({'error': 'Email already registered'}), 400
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        session['user_id'] = new_user.id
        session.permanent = True
        return jsonify({'message': 'Registration successful', 'username': new_user.username, 'preferences': new_user.preferences}), 201
    except Exception as e:
        db.session.rollback()
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@main_bp.route('/login', methods=['POST'])
def login():
    logger = logging.getLogger('app')
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
        user = User.query.filter(
            (func.lower(User.username) == identifier.lower()) | 
            (func.lower(User.email) == identifier.lower())
        ).first()
        if not user:
            return jsonify({'error': 'User not found. Please register or check your username/email.'}), 401
        if user.password != password:
            return jsonify({'error': 'Incorrect password. Please try again.'}), 401
        session['user_id'] = user.id
        session.permanent = True
        return jsonify({
            'message': 'Login successful',
            'username': user.username,
            'preferences': user.preferences,
            'joinDate': user.created_at.strftime('%B %d, %Y') if user.created_at else 'Unknown'
        })
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed due to an unexpected error'}), 500

@main_bp.route('/logout', methods=['POST'])
def logout():
    logger = logging.getLogger('app')
    try:
        ip_address = get_remote_address()
        if not check_rate_limit(f"logout:{ip_address}", 10, 60):
            return jsonify({'error': 'Rate limit exceeded: 10 per minute'}), 429
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        session.clear()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({'error': 'Logout failed'}), 500

@main_bp.route('/profile', methods=['GET'])
def profile():
    logger = logging.getLogger('app')
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        user = db.session.get(User, session['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
        return jsonify({
            'username': user.username,
            'preferences': user.preferences,
            'joinDate': user.created_at.strftime('%B %d, %Y') if user.created_at else 'Unknown'
        })
    except Exception as e:
        logger.error(f"Profile error: {str(e)}")
        return jsonify({'error': 'Failed to fetch profile'}), 500

@main_bp.route('/preferences', methods=['POST'])
def update_preferences():
    logger = logging.getLogger('app')
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
        return jsonify({'message': 'Preferences updated'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Preferences error: {str(e)}")
        return jsonify({'error': 'Failed to update preferences'}), 500

@main_bp.route('/summarize', methods=['POST'])
def summarize():
    logger = logging.getLogger('app')
    try:
        ip_address = get_remote_address()
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
            
        return jsonify({'summary': summary})
        
    except ValueError as ve:
        logger.error(f"Summarization failed: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Summarization failed unexpectedly: {str(e)}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during summarization. Please try again.'}), 500

@main_bp.route('/analyze', methods=['POST'])
def analyze():
    logger = logging.getLogger('app')
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
        analysis = analyze_text(text)
        if 'user_id' not in session:
            session['trial_used'] = True
        return jsonify({'analysis': analysis})
    except ValueError as ve:
        logger.error(f"Analyze error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Analyze error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred during analysis'}), 500

@main_bp.route('/extract', methods=['POST'])
async def extract():
    logger = logging.getLogger('app')
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
        content = await fetch_url_content(url)
        if 'user_id' not in session:
            session['trial_used'] = True
        return jsonify({'content': content})
    except Exception as e:
        logger.error(f"Extract error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@main_bp.route('/keywords', methods=['POST'])
def keywords():
    logger = logging.getLogger('app')
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
        keywords_list = extract_keywords(text)
        if 'user_id' not in session:
            session['trial_used'] = True
        return jsonify({'keywords': keywords_list})
    except Exception as e:
        logger.error(f"Keywords error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@main_bp.route('/sentiment', methods=['POST'])
def sentiment():
    logger = logging.getLogger('app')
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
        sentiment_result = analyze_sentiment(text)
        if 'user_id' not in session:
            session['trial_used'] = True
        return jsonify({'sentiment': sentiment_result})
    except Exception as e:
        logger.error(f"Sentiment error: {str(e)}")
        return jsonify({'error': str(e)}), 400
