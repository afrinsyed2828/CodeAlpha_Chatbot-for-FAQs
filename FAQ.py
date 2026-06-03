import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime

# Check and import dependencies with error handling
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError as e:
    NLTK_AVAILABLE = False
    st.error(f"NLTK is not installed. Error: {e}")
    st.info("Please run: pip install nltk")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    st.error(f"scikit-learn is not installed. Error: {e}")
    st.info("Please run: pip install scikit-learn")

# Download required NLTK data only if NLTK is available
if NLTK_AVAILABLE:
    @st.cache_resource
    def download_nltk_data():
        """Download necessary NLTK datasets"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
    
    # Download NLTK data
    download_nltk_data()
    
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
else:
    lemmatizer = None
    stop_words = set()

# Custom CSS for modern UI
def load_css():
    """Load custom CSS for the chatbot interface"""
    st.markdown("""
    <style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat message container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* User message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 70%;
        float: right;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        animation: slideInRight 0.3s ease-out;
    }
    
    /* Bot message styling */
    .bot-message {
        background: white;
        color: #333;
        padding: 12px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 70%;
        float: left;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* Timestamp styling */
    .timestamp {
        font-size: 10px;
        color: #999;
        margin-top: 5px;
    }
    
    /* Score badge styling */
    .score-badge {
        background: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        display: inline-block;
        margin-left: 10px;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideInLeft {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    /* Clear fix for floated elements */
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
    
    /* Header styling */
    .header {
        text-align: center;
        padding: 20px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    .header h1 {
        color: white;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    
    .header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1em;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 30px;
        border-radius: 25px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .user-message, .bot-message {
            max-width: 85%;
        }
        
        .header h1 {
            font-size: 1.8em;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Text preprocessing functions
def clean_text(text):
    """Clean and preprocess text"""
    if not NLTK_AVAILABLE:
        # Simple cleaning without NLTK
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return " ".join(text.split())
    
    text = str(text).lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Simple tokenization
    tokens = text.split()
    
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            cleaned_tokens.append(lemmatizer.lemmatize(token))
    
    return " ".join(cleaned_tokens)

# Load FAQ data
@st.cache_data
def load_faq_data():
    """Load FAQ data from CSV file"""
    try:
        df = pd.read_csv(
            "faq_data.csv",
            encoding="utf-8",
            engine="python",
            on_bad_lines="skip"
        )
        
        required_columns = ["question", "answer"]
        
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing column: {col}")
                return None
        
        df = df.dropna(subset=["question", "answer"])
        return df
        
    except FileNotFoundError:
        st.warning("faq_data.csv not found. Using sample data.")
        return None
    except Exception as e:
        st.error(f"Error loading FAQ data: {e}")
        return None

# Initialize TF-IDF vectorizer and FAQ vectors
@st.cache_resource
def initialize_vectorizer(df):
    """Initialize TF-IDF vectorizer and transform FAQ questions"""
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is required but not available")
        return None, None, None
    
    if df is None or df.empty:
        return None, None, None
    
    # Preprocess all FAQ questions
    df['processed_question'] = df['question'].apply(clean_text)
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Create TF-IDF vectors for FAQ questions
    faq_vectors = vectorizer.fit_transform(df['processed_question'])
    
    return vectorizer, faq_vectors, df

# Find best matching answer
def find_best_answer(user_question, vectorizer, faq_vectors, df, threshold=0.3):
    """Find the best matching FAQ answer for the user's question using cosine similarity"""
    if not SKLEARN_AVAILABLE:
        return None, None, 0.0
    
    # Preprocess user question
    processed_user_q = clean_text(user_question)
    
    # Convert user question to TF-IDF vector
    user_vector = vectorizer.transform([processed_user_q])
    
    # Calculate cosine similarity with all FAQ questions
    similarities = cosine_similarity(user_vector, faq_vectors).flatten()
    
    # Find best match
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    if best_score >= threshold:
        matched_question = df.iloc[best_match_idx]['question']
        answer = df.iloc[best_match_idx]['answer']
        return answer, matched_question, best_score
    else:
        return None, None, best_score

# Display message with animation
def display_message_with_animation(message, is_user, matched_q=None, score=None):
    """Display message with typing animation and styling"""
    container = st.container()
    
    if is_user:
        with container:
            st.markdown(f"""
            <div class="user-message">
                {message}
                <div class="timestamp">{datetime.now().strftime("%I:%M %p")}</div>
            </div>
            <div class="clearfix"></div>
            """, unsafe_allow_html=True)
    else:
        # Typing animation placeholder
        with st.spinner('Bot is typing...'):
            time.sleep(0.5)
        
        with container:
            score_html = f'<span class="score-badge">Match: {score:.2%}</span>' if score else ''
            matched_html = f'<br><small>📚 Matched: {matched_q}</small>' if matched_q else ''
            
            st.markdown(f"""
            <div class="bot-message">
                {message}
                {score_html}
                {matched_html}
                <div class="timestamp">{datetime.now().strftime("%I:%M %p")}</div>
            </div>
            <div class="clearfix"></div>
            """, unsafe_allow_html=True)

# Main app
def main():
    """Main function to run the Streamlit app"""
    
    # Check if required libraries are available
    if not SKLEARN_AVAILABLE:
        st.error("""
        ## ⚠️ Missing Required Library
        
        The `scikit-learn` library is not installed. This is required for the chatbot to function.
        
        ### To fix this issue:
        
        1. Make sure you have a `requirements.txt` file with:
