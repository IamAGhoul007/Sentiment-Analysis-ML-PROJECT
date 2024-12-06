import streamlit as st
import pandas as pd
import sqlite3
import base64
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
import bcrypt

# Database connection function with caching
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

conn = get_db_connection()
c = conn.cursor()

# Create database tables if they don’t exist
def create_tables():
    with conn:
        c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, profile_pic TEXT)')
        c.execute('''CREATE TABLE IF NOT EXISTS sentiment_analysis_results (
                     id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, text TEXT, sentiment TEXT, 
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

create_tables()

# Initialize session state
if 'username' not in st.session_state:
    st.session_state['username'] = 'Guest'
if 'profile_pic' not in st.session_state:
    st.session_state['profile_pic'] = "/Users/aaditya/Desktop/ML_Project copy/a.webp"

# User functions
def signup(username, password, profile_pic):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        with conn:
            conn.execute('INSERT INTO users (username, password, profile_pic) VALUES (?, ?, ?)', 
                         (username, hashed_password.decode('utf-8'), profile_pic))
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login(username, password):
    with conn:
        user = conn.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        if user and bcrypt.checkpw(password.encode('utf-8'), user[1].encode('utf-8')):
            st.session_state['username'] = username
            st.session_state['profile_pic'] = user[2] if user[2] else "/Users/aaditya/Desktop/ML_Project copy/a.webp"
            return True
    return False
    
# Display trainer card
def display_trainer_card():
    st.sidebar.image(st.session_state['profile_pic'], use_container_width=True)
    st.sidebar.title(f"{st.session_state['username'].capitalize()} - Sentiment Analyst")

# Load data with caching to avoid reloading
@st.cache_data
def load_data(file_path):
    column_names = ['sentiment', 'id', 'date', 'flag', 'user', 'text']
    data = pd.read_csv(file_path, header=None, names=column_names, encoding='ISO-8859-1')
    return data

# Preprocess and split data, then cache the result to avoid recomputation
@st.cache_data
def preprocess_data(data):
    sentiment_map = {0: 0, 4: 1}
    data['sentiment_numeric'] = data['sentiment'].map(sentiment_map)
    data = data.dropna(subset=['sentiment_numeric', 'text'])

    X = data['text']
    y = data['sentiment_numeric']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train and cache model and vectorizer for reuse
@st.cache_resource
def train_model(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train)

    y_pred = classifier.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['negative', 'positive'])

    return classifier, vectorizer, accuracy, report

# Perform prediction
def make_prediction(classifier, vectorizer, text):
    text_vectorized = vectorizer.transform([text])
    prediction = classifier.predict(text_vectorized)
    return 'positive' if prediction[0] == 1 else 'negative'

# Main app interface
def main_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Login", "Sign Up", "Continue as Guest", "Sentiment Analysis", "Metrics"])

    if page == "Login":
        login_page()
    elif page == "Sign Up":
        signup_page()
    elif page == "Continue as Guest":
        continue_as_guest_page()
    elif page == "Sentiment Analysis":
        display_trainer_card()
        sentiment_analysis_page()
    elif page == "Metrics":
        display_metrics_page()

# Login page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.success(f"Welcome back, {st.session_state['username']}!")
        else:
            st.error("Invalid username or password.")

# Sign Up page
def signup_page():
    st.title("Sign Up")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")

    # Profile picture selection
    st.subheader("Choose a Profile Picture")
    default_picture = "/Users/aaditya/Desktop/ML_Project copy/a.webp"
    uploaded_file = st.file_uploader("Upload your own picture (optional)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to a base64 string
        profile_pic = f"data:image/jpeg;base64,{base64.b64encode(uploaded_file.read()).decode()}"
    else:
        profile_pic = default_picture

    if st.button("Sign Up"):
        if signup(username, password, profile_pic):
            st.success(f"Account created successfully. Welcome, {username}!")
            st.session_state['username'] = username
            st.session_state['profile_pic'] = profile_pic
        else:
            st.error("Username already taken. Please choose another.")

# Continue as Guest page
def continue_as_guest_page():
    st.title("Continue as Guest")
    st.session_state['username'] = "Guest"
    st.info("You are continuing as Guest.")

# Sentiment analysis interface
def sentiment_analysis_page():
    st.markdown("<h1 class='big-font'>Sentiment Analyser</h1>", unsafe_allow_html=True)
    st.write("Welcome to the world of Sentiment Analysis!")

    data = load_data('training.1600000.processed.noemoticon.csv')
    if data is None:
        st.error("Failed to load data.")
        return

    X_train, X_test, y_train, y_test = preprocess_data(data)
    classifier, vectorizer, accuracy, report = train_model(X_train, X_test, y_train, y_test)
    if classifier is None or vectorizer is None:
        st.error("Model training failed.")
        return

    text = st.text_area("Enter a piece of text to analyze:")
    if st.button("Analyze Sentiment"):
        if text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                prediction = make_prediction(classifier, vectorizer, text)
            st.success("Analysis complete!")
            st.markdown(f"<h2>Sentiment: {prediction}</h2>", unsafe_allow_html=True)

            store_sentiment_analysis_results(st.session_state['username'], text, prediction)
            display_previous_results()

# Store sentiment analysis results in the database
def store_sentiment_analysis_results(username, text, sentiment):
    with conn:
        conn.execute('INSERT INTO sentiment_analysis_results (username, text, sentiment, timestamp) VALUES (?, ?, ?, ?)', 
                     (username, text, sentiment, datetime.now()))
        conn.commit()

# Display previous sentiment analysis results
def display_previous_results():
    previous_results = conn.execute('SELECT text, sentiment, timestamp FROM sentiment_analysis_results WHERE username=? ORDER BY timestamp DESC', (st.session_state['username'],)).fetchall()
    if previous_results:
        st.write("Previous Sentiment Analysis Results:")
        results_df = pd.DataFrame(previous_results, columns=['Text', 'Sentiment', 'Timestamp'])
        st.write(results_df)

# Display metrics page with a bar chart and metrics for logged-in user
def display_metrics_page():
    st.title("Metrics")

    previous_results = conn.execute('SELECT sentiment FROM sentiment_analysis_results WHERE username=?', (st.session_state['username'],)).fetchall()
    if previous_results:
        sentiment_counts = {'positive': 0, 'negative': 0}
        for result in previous_results:
            if result[0] == 'positive':
                sentiment_counts['positive'] += 1
            elif result[0] == 'negative':
                sentiment_counts['negative'] += 1

        st.subheader("Sentiment Counts")
        sentiment_counts_df = pd.DataFrame.from_dict(sentiment_counts, orient='index', columns=['Count'])
        st.bar_chart(sentiment_counts_df)

        st.subheader("Analysis Metrics")
        st.metric(label="Total Positive Sentiments", value=sentiment_counts['positive'])
        st.metric(label="Total Negative Sentiments", value=sentiment_counts['negative'])
    else:
        st.info("No sentiment analysis results to display for this user.")

# Run the app
if __name__ == "__main__":
    main_app()
