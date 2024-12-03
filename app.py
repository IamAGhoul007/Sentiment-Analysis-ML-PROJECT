import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
import sqlite3
from datetime import datetime

# Database setup
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

def create_users_table():
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, profile_pic TEXT)''')
    conn.commit()

def create_sentiment_analysis_results_table():
    c.execute('''CREATE TABLE IF NOT EXISTS sentiment_analysis_results (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, text TEXT, sentiment TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()

create_users_table()
create_sentiment_analysis_results_table()

# Initialize session state for username and profile pic if they don't exist
if 'username' not in st.session_state:
    st.session_state['username'] = 'Guest'
if 'profile_pic' not in st.session_state:
    st.session_state['profile_pic'] = "/Users/aaditya/Desktop/ML_Project/a.webp"  # Default profile picture

# Function to handle user login
def login(username, password):
    c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
    user = c.fetchone()
    if user:
        st.session_state['username'] = username
        st.session_state['profile_pic'] = user[2] if user[2] else "/Users/aaditya/Desktop/ML_Project/a.webp"
        return True
    return False

# Function to handle user signup
def signup(username, password):
    try:
        c.execute('INSERT INTO users (username, password, profile_pic) VALUES (?, ?, ?)', 
                  (username, password, "/Users/aaditya/Desktop/ML_Project/a.webp"))  # Default profile pic
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# Function to update profile pic
def update_profile_pic(username, new_pic_url):
    c.execute('UPDATE users SET profile_pic=? WHERE username=?', (new_pic_url, username))
    conn.commit()
    st.session_state['profile_pic'] = new_pic_url

# Function to store sentiment analysis results in the database with accurate timestamp
def store_sentiment_analysis_results(username, text, sentiment):
    # Get the current timestamp
    current_time = datetime.now()
    # Insert the sentiment analysis result with the captured timestamp
    c.execute('INSERT INTO sentiment_analysis_results (username, text, sentiment, timestamp) VALUES (?, ?, ?, ?)', 
              (username, text, sentiment, current_time))
    conn.commit()

# Function to get previous sentiment analysis results from database
def get_previous_sentiment_analysis_results(username):
    c.execute('SELECT id, text, sentiment, timestamp FROM sentiment_analysis_results WHERE username=? ORDER BY timestamp DESC', (username,))
    results = c.fetchall()
    return results

# Login/Sign Up Popup
def login_signup_popup():
    login_option = st.sidebar.radio("Login or Sign Up", ["Login", "Sign Up", "Continue as Guest"])
    if login_option == "Login":
        st.sidebar.subheader("Login to your account")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if login(username, password):
                st.sidebar.success(f"Welcome back, {st.session_state['username']}!")
            else:
                st.sidebar.error("Invalid username or password.")
    elif login_option == "Sign Up":
        st.sidebar.subheader("Create a new account")
        username = st.sidebar.text_input("New Username")
        password = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Sign Up"):
            if signup(username, password):
                st.sidebar.success(f"Account created successfully. Welcome, {username}!")
                st.session_state['username'] = username
            else:
                st.sidebar.error("Username already taken. Please choose another.")
    else:
        st.session_state['username'] = "Guest"
        st.sidebar.info("You're logged in as Guest.")

# Edit profile popup
def edit_profile_popup():
    st.sidebar.subheader("Edit Profile")
    new_profile_pic = st.sidebar.text_input("New Profile Picture URL")
    if st.sidebar.button("Update Profile Picture"):
        update_profile_pic(st.session_state['username'], new_profile_pic)
        st.sidebar.success("Profile picture updated!")

# Trainer Card Sidebar
def display_trainer_card():
    st.sidebar.markdown(f"<h2>Trainer Card</h2>", unsafe_allow_html=True)
    st.sidebar.image(st.session_state['profile_pic'], use_column_width=True)
    st.sidebar.markdown(f"<h3>{st.session_state['username'].capitalize()}</h3>", unsafe_allow_html=True)
    st.sidebar.markdown("Sentiment Analyst Extraordinaire")
    if st.session_state['username'] != "Guest":
        if st.sidebar.button("Edit Profile"):
            edit_profile_popup()

# Load the Twitter training data
def load_data(file_path):
    column_names = ['sentiment', 'id', 'date', 'flag', 'user', 'text']
    try:
        data = pd.read_csv(file_path, header=None, names=column_names, encoding='ISO-8859-1')
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Preprocess the data
def preprocess_data(data):
    try:
        sentiment_map = {0: 0, 4: 1}
        data['sentiment_numeric'] = data['sentiment'].map(sentiment_map)
        data = data.dropna(subset=['sentiment_numeric', 'text'])

        X = data['text']
        y = data['sentiment_numeric']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        raise

# Train a Naive Bayes classifier
def train_model(X_train, X_test, y_train, y_test):
    try:
        vectorizer = TfidfVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        classifier = MultinomialNB()
        classifier.fit(X_train_vectorized, y_train)

        y_pred = classifier.predict(X_test_vectorized)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['negative', 'positive'])

        st.write(f"Model Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(report)

        return classifier, vectorizer, accuracy, report
    except Exception as e:
        st.error("Error training model: " + str(e))
        return None, None, None, None

# Function to make a prediction
def make_prediction(classifier, vectorizer, text):
    text_vectorized = vectorizer.transform([text])
    prediction = classifier.predict(text_vectorized)
    return 'positive' if prediction[0] == 1 else 'negative'

# Main App
def main_app():

    login_signup_popup()
    display_trainer_card()

    st.markdown("<h1 class='big-font'>Sentiment Analyser</h1>", unsafe_allow_html=True)
    st.write("Welcome to the world of Sentiment Analysis!")

    # Load the Twitter training data
    data = load_data('training.1600000.processed.noemoticon.csv')

    if data is None:
        st.error("Failed to load data.")
        return

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    if X_train is None:
        st.error("Data preprocessing failed. Please check your data and try again.")
        return

    # Train a Naive Bayes classifier
    classifier, vectorizer, accuracy, report = train_model(X_train, X_test, y_train, y_test)

    if classifier is None or vectorizer is None:
        st.error("Model training failed. Please check your code and try again.")
        return

    # Get user input
    text = st.text_area("Enter a piece of text to analyze:")

    # Make a prediction
    if st.button("Analyze Sentiment"):
        if text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                prediction = make_prediction(classifier, vectorizer, text)
            st.success("Analysis complete!")
            st.markdown(f"<h2>Sentiment: {prediction}</h2>", unsafe_allow_html=True)

            # Store sentiment analysis
            store_sentiment_analysis_results(st.session_state['username'], text, prediction)

            # Display previous sentiment analysis results
            previous_results = get_previous_sentiment_analysis_results(st.session_state['username'])

            if previous_results:
                st.write("Previous Sentiment Analysis Results:")
                results_df = pd.DataFrame(previous_results, columns=['ID', 'Text', 'Sentiment', 'Timestamp'])
                st.write(results_df)

                # Create sentiment counts
                sentiment_counts = {'Positive': 0, 'Negative': 0}

                for result in previous_results:
                    if result[2] == 'positive':
                        sentiment_counts['Positive'] += 1
                    elif result[2] == 'negative':
                        sentiment_counts['Negative'] += 1

                # Convert dictionary to DataFrame
                sentiment_counts_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])

                # Plot sentiment counts as a bar chart
                st.bar_chart(sentiment_counts_df.set_index('Sentiment'))

# Run the app
if _name_ == "_main_":
    main_app()
