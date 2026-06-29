import sqlite3
import bcrypt

conn = sqlite3.connect("users.db", check_same_thread=False)
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

def add_user(username, password):
    hashed = bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    ).decode()

    c.execute(
        "INSERT INTO users(username,password) VALUES(?,?)",
        (username, hashed)
    )
    conn.commit()

def login_user(username, password):

    c.execute(
        "SELECT * FROM users WHERE username=?",
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

def save_feedback(username, review, sentiment):
    c.execute(
        "INSERT INTO feedback(username,review,sentiment) VALUES(?,?,?)",
        (username, review, sentiment)
    )
    conn.commit()