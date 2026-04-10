import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("FA-KES-Dataset.csv", encoding="latin-1")

# Prepare data
data["news"] = data["article_title"] + " " + data["article_content"]

# Features & Labels
x = data["news"]
y = data["labels"]

# Vectorizer
vectorizer = CountVectorizer()
x_vectorized = vectorizer.fit_transform(x)

# Model
model = MultinomialNB()
model.fit(x_vectorized, y)

# Save model (run once, then comment these lines if needed)
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))


# 🔥 Prediction Function (Flask will use this)
def predict_news(text):
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]

    # Confidence score 🔥
    prob = model.predict_proba(transformed_text)[0]

    confidence = round(max(prob) * 100, 2)
    print(prob)
    print(confidence)
    if prediction == 1:
        return "Real News ✅", confidence
    else:
        return "Fake News ❌", confidence
predict_news("trum has been start attack on saudi")