import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# Load the pre-trained model
personality_model = load_model('final_personality_detection_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    # Tokenize the text data and create sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)

    # Pad sequences to have a fixed length
    max_length = 101
    X = pad_sequences(sequences, maxlen=max_length)

    #sequences = word_tokenize(data)
    #max_length = max(len(sequence) for sequence in sequences)
    #X = pad_sequences(sequences, maxlen=max_length)
    predicted_traits = personality_model.predict(X)

    # Calculate sentiment scores using VADER
    sentiment_scores = calculate_sentiment_scores(data)

    # Prepare response
    response = {
        'predictions': predicted_traits.tolist(),
        'sentiment_scores': sentiment_scores
    }

    return jsonify(response)

def calculate_sentiment_scores(data):
    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    sentiment_scores = []

    # Calculate sentiment score for each sentence
    for sentence in data:
        sentiment = sid.polarity_scores(sentence)
        sentiment_scores.append(sentiment['compound'])  # Compound score represents overall sentiment

    return sentiment_scores

if __name__ == '__main__':
    app.run(debug=True)
