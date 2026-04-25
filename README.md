# Summery AI

Summery AI is a Flask-based artificial intelligence web application designed to help users quickly summarize, analyze, and extract key information from text or URLs. 

## Features
- **Text Summarization:** Uses Google Gemini via LangChain to intelligently reduce large blocks of text into concise summaries.
- **Sentiment Analysis:** Analyzes the sentiment of a given text (Positive/Negative/Neutral) using Gemini.
- **Keyword Extraction:** Identifies the top keywords in an article using Gemini.
- **URL Content Extraction:** Fetches and cleans textual content from web pages to allow direct analysis.
- **User Authentication:** Accounts backed by an SQLite database with rate-limiting and user preferences (summary lengths).

## Setup Instructions

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Set up environment variables:
   Copy `.env.example` to `.env` and add your `GOOGLE_API_KEY`.
6. Run the application:
   ```bash
   python app.py
   ```
7. The app will be available at `http://localhost:5000/`.

## Technology Stack
- **Backend:** Python, Flask, Waitress
- **NLP:** LangChain, Google Gemini API
- **Database:** SQLAlchemy, SQLite
- **Networking:** aiohttp, BeautifulSoup4
