import streamlit as st
import pandas as pd
import sqlite3
import pickle
import bcrypt
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

nltk.download("stopwords")

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================
# LOAD FILES
# =====================================

model = pickle.load(
    open("sentiment_model.pkl", "rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl", "rb")
)

label_encoder = pickle.load(
    open("label_encoder.pkl", "rb")
)

try:
    cm = pickle.load(
        open("cm.pkl", "rb")
    )
except:
    cm = None

# =====================================
# DATABASE
# =====================================

conn = sqlite3.connect(
    "users.db",
    check_same_thread=False
)

c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT UNIQUE,
password TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS feedback(
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT,
review TEXT,
sentiment TEXT
)
""")

conn.commit()

# =====================================
# NLP PREPROCESSING
# =====================================

ps = PorterStemmer()

stop_words = set(
    stopwords.words("english")
)

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z ]',
        '',
        text
    )

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# =====================================
# USER FUNCTIONS
# =====================================

def add_user(username, password):

    hashed = bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    ).decode()

    c.execute(
        """
        INSERT INTO users
        (username,password)
        VALUES(?,?)
        """,
        (username, hashed)
    )

    conn.commit()

def login_user(username, password):

    c.execute(
        """
        SELECT *
        FROM users
        WHERE username=?
        """,
        (username,)
    )

    user = c.fetchone()

    if user:

        stored_password = user[2]

        if bcrypt.checkpw(
            password.encode(),
            stored_password.encode()
        ):
            return user

    return None

def save_feedback(
        username,
        review,
        sentiment
):

    c.execute(
        """
        INSERT INTO feedback
        (username,review,sentiment)
        VALUES(?,?,?)
        """,
        (
            username,
            review,
            sentiment
        )
    )

    conn.commit()

# =====================================
# SESSION
# =====================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# =====================================
# LOGIN PAGE
# =====================================

if not st.session_state.logged_in:

    st.title(
        "🔐 Sentiment Analysis System"
    )

    menu = st.selectbox(
        "Choose",
        ["Login", "Register"]
    )

    # REGISTER

    if menu == "Register":

        st.subheader(
            "Create Account"
        )

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        confirm = st.text_input(
            "Confirm Password",
            type="password"
        )

        if st.button("Register"):

            if password != confirm:

                st.error(
                    "Passwords do not match"
                )

            else:

                try:

                    add_user(
                        username,
                        password
                    )

                    st.success(
                        "Registration Successful"
                    )

                except:

                    st.error(
                        "Username already exists"
                    )

    # LOGIN

    else:

        st.subheader("Login")

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            user = login_user(
                username,
                password
            )

            if user:

                st.session_state.logged_in = True

                st.session_state.username = username

                st.success(
                    "Login Successful"
                )

                st.rerun()

            else:

                st.error(
                    "Invalid Credentials"
                )

# =====================================
# MAIN DASHBOARD
# =====================================

else:

    st.sidebar.title(
        "📊 Dashboard"
    )

    menu = st.sidebar.radio(
        "Navigation",
        [
            "Predict Sentiment",
            "Analytics Dashboard",
            "Feedback History",
            "Logout"
        ]
    )

    st.sidebar.success(
        f"User: {st.session_state.username}"
    )

    feedback_df = pd.read_sql(
        "SELECT * FROM feedback",
        conn
    )
    
# =====================================
# PREDICT SENTIMENT
# =====================================

if menu == "Predict Sentiment":

    st.title("😊 Sentiment Prediction")

    review = st.text_area(
        "Enter Review",
        height=200
    )

    if st.button("Predict Sentiment"):

        if review.strip() == "":

            st.warning(
                "Please enter a review"
            )

        else:

            processed = preprocess_text(
                review
            )

            vector = vectorizer.transform(
                [processed]
            )

            prediction = model.predict(
                vector
            )

            sentiment = (
                label_encoder
                .inverse_transform(
                    prediction
                )[0]
            )

            if sentiment.lower() == "positive":

                st.success(
                    "😊 Positive Sentiment"
                )

            elif sentiment.lower() == "negative":

                st.error(
                    "😞 Negative Sentiment"
                )

            else:

                st.info(
                    "😐 Neutral Sentiment"
                )

            save_feedback(
                st.session_state.username,
                review,
                sentiment
            )

            st.success(
                "Feedback Saved Successfully"
            )

# =====================================
# ANALYTICS DASHBOARD
# =====================================

elif menu == "Analytics Dashboard":

    st.title("📊 Analytics Dashboard")

    total_reviews = len(
        feedback_df
    )

    positive_count = len(
        feedback_df[
            feedback_df["sentiment"]
            .str.lower()
            == "positive"
        ]
    )

    negative_count = len(
        feedback_df[
            feedback_df["sentiment"]
            .str.lower()
            == "negative"
        ]
    )

    neutral_count = len(
        feedback_df[
            feedback_df["sentiment"]
            .str.lower()
            == "neutral"
        ]
    )

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Total Reviews",
        total_reviews
    )

    col2.metric(
        "Positive",
        positive_count
    )

    col3.metric(
        "Negative",
        negative_count
    )

    col4.metric(
        "Neutral",
        neutral_count
    )

    st.markdown("---")

    # ==========================
    # SENTIMENT DISTRIBUTION
    # ==========================

    st.subheader(
        "Sentiment Distribution"
    )

    sentiment_counts = (
        feedback_df["sentiment"]
        .value_counts()
    )

    st.bar_chart(
        sentiment_counts
    )

    # ==========================
    # PIE CHART
    # ==========================

    st.subheader(
        "Sentiment Percentage"
    )

    fig, ax = plt.subplots(
        figsize=(6,6)
    )

    ax.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct="%1.1f%%"
    )

    ax.axis("equal")

    st.pyplot(fig)

    # ==========================
    # WORD CLOUDS
    # ==========================

    st.markdown("---")

    col1,col2 = st.columns(2)

    # POSITIVE WORD CLOUD

    with col1:

        st.subheader(
            "Positive Word Cloud"
        )

        positive_reviews = " ".join(
            feedback_df[
                feedback_df["sentiment"]
                .str.lower()
                == "positive"
            ]["review"]
            .astype(str)
        )

        if positive_reviews:

            wc = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate(
                positive_reviews
            )

            fig, ax = plt.subplots()

            ax.imshow(wc)

            ax.axis("off")

            st.pyplot(fig)

    # NEGATIVE WORD CLOUD

    with col2:

        st.subheader(
            "Negative Word Cloud"
        )

        negative_reviews = " ".join(
            feedback_df[
                feedback_df["sentiment"]
                .str.lower()
                == "negative"
            ]["review"]
            .astype(str)
        )

        if negative_reviews:

            wc = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate(
                negative_reviews
            )

            fig, ax = plt.subplots()

            ax.imshow(wc)

            ax.axis("off")

            st.pyplot(fig)

    # ==========================
    # CONFUSION MATRIX
    # ==========================

    st.markdown("---")

    st.subheader(
        "Confusion Matrix"
    )

    if cm is not None:

        fig, ax = plt.subplots(
            figsize=(6,5)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        st.pyplot(fig)

    else:

        st.info(
            "cm.pkl not found"
        )

# =====================================
# FEEDBACK HISTORY
# =====================================

elif menu == "Feedback History":

    st.title(
        "📝 Feedback History"
    )

    if feedback_df.empty:

        st.info(
            "No feedback available"
        )

    else:

        st.dataframe(
            feedback_df.sort_values(
                by="id",
                ascending=False
            ),
            use_container_width=True
        )

# =====================================
# LOGOUT
# =====================================

elif menu == "Logout":

    st.session_state.logged_in = False

    st.session_state.username = ""

    st.success(
        "Logged Out Successfully"
    )

    st.rerun()