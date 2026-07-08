import streamlit as st
from keras.models import load_model
from keras.utils import pad_sequences
from keras.datasets import imdb


import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "imdb_sentiment_model.keras"
)

model = load_model(MODEL_PATH)

word_index = imdb.get_word_index()

max_features = 10000



def preprocess_text(text):
    words = text.lower().split()

    encoded = [1]  
    for word in words:
        if word in word_index and word_index[word] < max_features:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  

    padded = pad_sequences(
        [encoded],
        maxlen=500
    )

    return padded

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    if user_input:
        sentiment, score = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {score}")
    else:
        st.warning("Please enter a review.")