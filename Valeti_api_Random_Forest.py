import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import string
import numpy as np
from flask import Flask,jsonify,request
#import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
lemmatizer= WordNetLemmatizer()



# Load the trained model and label encoder
import pickle

# Load the model from file
with open('RF_model.pkl', 'rb') as f:
    log_reg = pickle.load(f)

model = log_reg
le = LabelEncoder()
le.classes_ = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']



def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]
    
    return " " .join(text)

def remove_stop_words(text):

    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    
    text = text.split()

    text=[y.lower() for y in text]
    
    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()
            
def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= lemmatization(sentence)
    return sentence


def get_sentences(paragraph):
    # Split the paragraph into sentences
    sentences = paragraph.split('.')

    # Remove the last empty element (if any)
    if sentences[-1] == '':
        sentences = sentences[:-1]

    # Remove leading/trailing whitespaces from each sentence
    sentences = [sentence.strip() for sentence in sentences]

    return sentences

app = Flask(__name__)

import numpy as np

@app.route('/predict_emotions/', methods=['POST'])
def predict_emotions_api():
    # Get the input paragraph from the request
    data = request.json['paragraph']

    # Get the list of sentences from the input paragraph
    sentences_copy = get_sentences(data)

    sentences=[]
    for sentence in sentences_copy:
        sentences.append(normalized_sentence(sentence))

    # Perform emotion prediction for each sentence
    emotions = model.predict(sentences)

    # Convert emotions to a NumPy array
    emotions = np.array(emotions)

    # Get unique emotion labels and their counts
    unique_emotions, emotion_counts = np.unique(emotions, return_counts=True)

    # Calculate emotion percentages
    total_emotions = len(emotions)
    emotion_percentages = emotion_counts / total_emotions * 100

    # Create a dictionary of emotions and percentages
    emotion_results = {
        emotion: percentage for emotion, percentage in zip(unique_emotions, emotion_percentages)
    }

    # Prepare the response
    response = {
        'emotion_percentages': emotion_results
    }

    # Return the response as JSON
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5002)
