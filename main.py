import pandas as pd
import re
import pickle
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# DOWNLOAD NLTK DATA
# ==========================================

nltk.download("stopwords")

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("train.csv", encoding="latin1")

# Keep only required columns
df = df[["text", "sentiment"]]

# Remove missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

print("Dataset Shape:", df.shape)

# ==========================================
# TEXT PREPROCESSING
# ==========================================

ps = PorterStemmer()

stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = [
        ps.stem(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

print("Preprocessing text...")

df["text"] = df["text"].apply(preprocess)

# ==========================================
# LABEL ENCODING
# ==========================================

le = LabelEncoder()

y = le.fit_transform(df["sentiment"])

print("Classes:", list(le.classes_))

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

print("Applying TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X = tfidf.fit_transform(df["text"])

print("Feature Shape:", X.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# MODEL COMPARISON
# ==========================================

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Naive Bayes":
        MultinomialNB(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "Linear SVM":
        LinearSVC()
}

results = []

best_model = None
best_acc = 0
best_name = ""

print("\n============================")
print("MODEL COMPARISON")
print("============================\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results.append([
        name,
        round(acc * 100, 2)
    ])

    print(
        f"{name} Accuracy: {round(acc*100,2)}%"
    )

    if acc > best_acc:

        best_acc = acc

        best_model = model

        best_name = name

# ==========================================
# SAVE COMPARISON RESULTS
# ==========================================

comparison_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy"]
)

comparison_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nModel Comparison Saved")

# ==========================================
# BEST MODEL EVALUATION
# ==========================================

print("\n============================")
print("BEST MODEL")
print("============================")

print("Selected Model:", best_name)

pred = best_model.predict(X_test)

print(
    "\nAccuracy:",
    round(best_acc * 100, 2),
    "%"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        pred,
        target_names=le.classes_
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    pred
)

pickle.dump(
    cm,
    open("cm.pkl", "wb")
)

print("Confusion Matrix Saved")

# Optional image save

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title(
    f"Confusion Matrix ({best_name})"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.close()

# ==========================================
# SAVE MODEL FILES
# ==========================================

pickle.dump(
    best_model,
    open("sentiment_model.pkl", "wb")
)

pickle.dump(
    tfidf,
    open("vectorizer.pkl", "wb")
)

pickle.dump(
    le,
    open("label_encoder.pkl", "wb")
)

print("\n============================")
print("FILES SAVED")
print("============================")

print("sentiment_model.pkl")
print("vectorizer.pkl")
print("label_encoder.pkl")
print("cm.pkl")
print("model_comparison.csv")
print("confusion_matrix.png")

# ==========================================
# SAMPLE TEST
# ==========================================

sample = "The service was excellent and I loved it."

sample = preprocess(sample)

sample_vector = tfidf.transform(
    [sample]
)

prediction = best_model.predict(
    sample_vector
)

sentiment = le.inverse_transform(
    prediction
)[0]

print("\nSample Prediction:")
print(sample)
print("Sentiment:", sentiment)

print("\nTraining Completed Successfully!")