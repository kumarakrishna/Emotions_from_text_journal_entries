import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional

from flask import Flask, jsonify, request

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load the trained model and label encoder
import pickle

# Load the model from file
with open('lstm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load GloVe word embeddings
embeddings_index = {}
embedding_dim = 200  # Update with the appropriate dimension for your GloVe vectors

glove_file_path = 'path_to_glove_file.txt'  # Provide the path to your GloVe file

with open(glove_file_path) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

# Define preprocessing functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text):
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def Removing_punctuations(text):
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.Text = df.Text.apply(lambda text: lower_case(text))
    df.Text = df.Text.apply(lambda text: remove_stop_words(text))
    df.Text = df.Text.apply(lambda text: Removing_numbers(text))
    df.Text = df.Text.apply(lambda text: Removing_punctuations(text))
    df.Text = df.Text.apply(lambda text: Removing_urls(text))
    df.Text = df.Text.apply(lambda text: lemmatization(text))
    return df

def normalized_sentence(sentence):
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = Removing_numbers(sentence)
    sentence = Removing_punctuations(sentence)
    sentence = Removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence

def get_sentences(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return sentences

def calculate_emotion_percentages(paragraph):
    sentences = get_sentences(paragraph)
    total_sentences = len(sentences)
    emotions = {}

    for sentence in sentences:
        sentence = normalized_sentence(sentence)
        sentence = tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=229, truncating='pre')
        
        # Apply GloVe embeddings
        embedding_matrix = np.zeros((sentence.shape[1], embedding_dim))
        for word_index in sentence[0]:
            word = tokenizer.index_word[word_index]
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[word_index] = embedding_vector
        
        result = le.inverse_transform(np.argmax(model.predict(embedding_matrix[np.newaxis, :]), axis=-1))[0]
        proba = np.max(model.predict(embedding_matrix[np.newaxis, :]))

        if result in emotions:
            emotions[result] += 1
        else:
            emotions[result] = 1

    percentages = {emotion: (count / total_sentences) * 100 for emotion, count in emotions.items()}
    return emotions, percentages


app = Flask(__name__)

@app.route('/predict_emotions/', methods=['POST'])
def predict_emotions_api():
    data = request.get_json()
    paragraph = data['paragraph']

    emotions, percentages = calculate_emotion_percentages(paragraph)

    emotion_results = {
        emotion: percentages[emotion] for emotion in emotions.keys()
    }

    # Prepare the response
    response = {
        'emotion_percentages': emotion_results
    }
    # Return the response as JSON
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
