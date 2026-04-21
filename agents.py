import os
import re
import bleach
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

# Initialize the language model
# Using the available gemini-2.5-flash model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# 1. Summarization Agent
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert copywriter. Summarize the following text in under {max_words} words. Return ONLY the summarized text, no introductions or explanations."),
    ("human", "{text}")
])
summarize_chain = summary_prompt | llm | StrOutputParser()

def summarize_text(text: str, max_length: int = 150) -> str:
    if not text or len(text.strip()) < 20:
        raise ValueError("Text too short for summarization (minimum 20 characters required)")
    
    sanitized_text = bleach.clean(text[:10000])
    try:
        summary = summarize_chain.invoke({"text": sanitized_text, "max_words": max_length})
        return summary.strip()
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        raise ValueError(f"Summarization failed: {str(e)}")

# 2. Text Analysis Logic (Python-based for accurate counts)
def analyze_text(text: str) -> dict:
    if not text or len(text.strip()) < 10:
        raise ValueError("Text too short for analysis (minimum 10 characters required)")
    
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = clean_text.split()
    if not words:
        raise ValueError("No valid words found in text")
        
    word_count = len(words)
    unique_words = len(set(words))
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
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

# 3. Keyword Extraction Agent
keywords_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the top {top_n} most important keywords from the following text. Return a JSON array of strings ONLY. For example: [\"ai\", \"flask\", \"python\"]"),
    ("human", "{text}")
])
keywords_chain = keywords_prompt | llm | JsonOutputParser()

def extract_keywords(text: str, top_n: int = 10) -> list:
    if not text or len(text.strip()) < 10:
        raise ValueError("Text too short for keyword extraction")
    try:
        words = keywords_chain.invoke({"text": text[:5000], "top_n": top_n})
        if isinstance(words, list):
            return words[:top_n]
        return []
    except Exception as e:
        logging.error(f"Keyword extraction error: {e}")
        raise ValueError(f"Keyword extraction failed: {str(e)}")

# 4. Sentiment Analysis Agent
class SentimentOutput(BaseModel):
    sentiment: str = Field(description="One of: Positive, Negative, Neutral")
    positive: float = Field(description="Score between 0 and 100")
    negative: float = Field(description="Score between 0 and 100")
    neutral: float = Field(description="Score between 0 and 100")
    compound: float = Field(description="Overall sentiment score between -1 and 1")

sentiment_parser = JsonOutputParser(pydantic_object=SentimentOutput)
sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze the sentiment of the text. \n{format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=sentiment_parser.get_format_instructions())

sentiment_chain = sentiment_prompt | llm | sentiment_parser

def analyze_sentiment(text: str) -> dict:
    if not text or len(text.strip()) < 10:
        raise ValueError("Text too short for sentiment analysis")
    try:
        result = sentiment_chain.invoke({"text": text[:5000]})
        return result
    except Exception as e:
        logging.error(f"Sentiment analysis error: {e}")
        raise ValueError(f"Sentiment analysis failed: {str(e)}")
