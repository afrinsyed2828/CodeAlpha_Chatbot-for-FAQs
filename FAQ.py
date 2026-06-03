import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
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

download_nltk_data()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Custom CSS for modern UI
def load_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
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
    
    .timestamp {
        font-size: 10px;
        color: #999;
        margin-top: 5px;
    }
    
    .score-badge {
        background: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        display: inline-block;
        margin-left: 10px;
    }
    
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
    
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
    
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
    
    .info-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
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

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            cleaned_tokens.append(lemmatizer.lemmatize(token))
    return " ".join(cleaned_tokens)

@st.cache_data
def load_faq_data():
    try:
        df = pd.read_csv("faq_data.csv", encoding="utf-8", engine="python", on_bad_lines="skip")
        required_columns = ["question", "answer"]
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing column: {col}")
                return None
        df = df.dropna(subset=["question", "answer"])
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading FAQ data: {e}")
        return None

@st.cache_resource
def initialize_vectorizer(df):
    if df is None or df.empty:
        return None, None, None
    
    df['processed_question'] = df['question'].apply(clean_text)
    vectorizer = TfidfVectorizer()
    faq_vectors = vectorizer.fit_transform(df['processed_question'])
    return vectorizer, faq_vectors, df

def find_best_answer(user_question, vectorizer, faq_vectors, df, threshold=0.3):
    processed_user_q = clean_text(user_question)
    user_vector = vectorizer.transform([processed_user_q])
    similarities = cosine_similarity(user_vector, faq_vectors).flatten()
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    if best_score >= threshold:
        matched_question = df.iloc[best_match_idx]['question']
        answer = df.iloc[best_match_idx]['answer']
        return answer, matched_question, best_score
    else:
        return None, None, best_score

def display_message_with_animation(message, is_user, matched_q=None, score=None):
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

def main():
    load_css()
    
    # Header
    st.markdown('<div class="header"><h1>🤖 FAQ Chatbot Assistant</h1><p>Your AI-powered question answering system</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="info-card"><h3>📊 About</h3><p>This chatbot uses Natural Language Processing to answer your questions based on a FAQ database.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-card"><h3>⚙️ How it works</h3><p>1. Text preprocessing with NLTK<br>2. TF-IDF vectorization<br>3. Cosine similarity matching<br>4. Returns best matching answer</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-card"><h3>💡 Sample Questions</h3><p>• How do I reset my password?<br>• What is two-factor authentication?<br>• How to update my email address?<br>• What are your business hours?<br>• How to delete my account?</p></div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Load FAQ data
    df = load_faq_data()
    
    if df is None:
        sample_data = pd.DataFrame({
            'question': [
                'How do I reset my password?',
                'What is two-factor authentication?',
                'How to update my email address?',
                'What are your business hours?',
                'How to delete my account?',
                'What is your return policy?',
                'How can I contact support?',
                'Do you offer discounts for students?'
            ],
            'answer': [
                'To reset your password, click on "Forgot Password" link on the login page. You will receive an email with reset instructions.',
                'Two-factor authentication (2FA) adds an extra layer of security. You will need to verify your identity using a second method like SMS or authenticator app.',
                'To update your email address, go to Account Settings > Profile Information > Email Address. Click Edit and enter your new email.',
                'Our business hours are Monday to Friday, 9:00 AM to 6:00 PM EST. We are closed on weekends and major holidays.',
                'To delete your account, please contact our support team. They will guide you through the account deletion process.',
                'We offer a 30-day return policy for all unused items in original packaging. Please contact customer service to initiate a return.',
                'You can contact our support team via email at support@example.com or call us at 1-800-123-4567.',
                'Yes, we offer a 15% student discount with valid student ID. Contact our support team for more details.'
            ]
        })
        st.info("💡 Using sample FAQ data. Create your own 'faq_data.csv' file to customize!")
        df = sample_data
    
    vectorizer, faq_vectors, df = initialize_vectorizer(df)
    
    if vectorizer is None:
        st.error("Failed to initialize chatbot components.")
        return
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "bot",
                "content": "Hello! 👋 I'm your FAQ assistant. Ask me anything from our knowledge base!",
                "timestamp": datetime.now()
            }
        ]
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                display_message_with_animation(message["content"], True)
            else:
                display_message_with_animation(
                    message["content"], 
                    False, 
                    message.get("matched_q"), 
                    message.get("score")
                )
    
    # User input
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your question here:",
            key="user_input",
            placeholder="Ask me anything about our services...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    if send_button and user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        answer, matched_q, score = find_best_answer(user_input, vectorizer, faq_vectors, df)
        
        if answer:
            bot_response = answer
            bot_data = {
                "role": "bot",
                "content": bot_response,
                "matched_q": matched_q,
                "score": score,
                "timestamp": datetime.now()
            }
        else:
            bot_response = f"Sorry, I could not find a relevant answer. 😔\n\nPlease try rephrasing your question or contact support for assistance.\n\n(Confidence score: {score:.2%})"
            bot_data = {
                "role": "bot",
                "content": bot_response,
                "timestamp": datetime.now()
            }
        
        st.session_state.messages.append(bot_data)
        st.rerun()
    
    # Footer
    st.markdown('<div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.7);"><small>Powered by NLP • TF-IDF • Cosine Similarity</small></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
