import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

# Preprocessing function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return filtered_tokens

# Intents and patterns
intents = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "farewell": ["bye", "goodbye", "see you", "take care"],
    "status": ["how are you", "how do you do", "how's it going", "how are you doing"],
    "name": ["what is your name", "who are you"],
    "time": ["what time is it", "tell me the time"],
    "date": ["what is the date today", "what's the date", "tell me the date"],
    "weather": ["what's the weather", "how's the weather", "weather forecast"],
    "location": ["where am i", "what's my location", "current location"],
    "joke": ["tell me a joke", "make me laugh"],
    "thank_you": ["thank you", "thanks"]
}

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer().fit([
    item for sublist in intents.values() for item in sublist
    ])

# Intent identification function using cosine similarity
def identify_intent(user_input):
    user_input = ' '.join(preprocess(user_input))
    user_vec = vectorizer.transform([user_input])
    
    max_similarity = 0
    matched_intent = "default"
    for intent, patterns in intents.items():
        intent_vec = vectorizer.transform(patterns)
        similarity = cosine_similarity(user_vec, intent_vec).flatten()
        if max(similarity) > max_similarity:
            max_similarity = max(similarity)
            matched_intent = intent
    
    st.write(f"Matched intent: {matched_intent} with similarity: {max_similarity}")  # Debugging statement
    return matched_intent

# Entity extraction function
def extract_entities(user_input):
    doc = nlp(user_input)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

# Chatbot response function
def chatbot_response(user_input):
    preprocessed_input = preprocess(user_input)
    intent = identify_intent(' '.join(preprocessed_input))
    entities = extract_entities(user_input)

    responses = {
        "greeting": "Hi there! How can I help you?",
        "farewell": "Goodbye! Have a nice day!",
        "status": "I'm just a bot, but I'm doing great! How about you?",
        "name": "I'm your friendly chatbot, here to assist you.",
        "time": "I'm not equipped to tell the time yet, but you can check your device!",
        "date": "I'm not equipped to tell the date yet, but you can check your calendar!",
        "weather": "I'm not equipped to provide weather updates yet, but you can check a weather app!",
        "location": "I'm not equipped to provide location services yet.",
        "joke": "Why don't scientists trust atoms? Because they make up everything!",
        "thank_you": "You're welcome!",
        "default": "I'm sorry, I don't understand that. Can you please rephrase?"
    }

    return responses.get(intent, responses["default"])

# Streamlit app
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .chatbox {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .user_message {
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .bot_message {
        background-color: #fff9c4;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🤖 Rule-Based Chatbot with NLP")
st.write("Type your message below and press Enter:")

user_input = st.text_input("You:", key="user_input")

if user_input:
    response = chatbot_response(user_input.lower())
    st.markdown(f"<div class='chatbox'><div class='user_message'><b>You:</b> {user_input}</div><div class='bot_message'><b>Bot:</b> {response}</div></div>", unsafe_allow_html=True)
