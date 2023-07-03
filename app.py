# from urllib.parse import quote_from_bytes
import os
import json
from  requests import get
from bs4 import BeautifulSoup
import nltk
from socket import *
import numpy as np
from flask import Flask, jsonify, render_template, request, make_response, redirect, url_for
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# from requests import get 
import requests

from werkzeug.utils import secure_filename

# import wikipedia
from googlesearch import search
# import time


lemmatizer = WordNetLemmatizer()


lemmatizer = WordNetLemmatizer()
app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploaded'
# file=os.path.join(os.getcwd(),'uploaded_file')
# print(file,"files")
# app.config['UPLOAD_FOLDER'] = file

client =MongoClient('mongodb://localhost:27017')
app.config['SECRET_KEY']='thisissecret'
db = client['bot']
collection = db['data2']
# collection2 = db['files']
# userText=[]

with open("C:/Users/hp pc/OneDrive/Desktop/dchat/data.json") as file:
        data = json.load(file) 

lemmatizer = WordNetLemmatizer()
# tfidf_vectorizer = TfidfVectorizer(stop_words='english')
question = []
tags = []

for intent in data['intents']:
    for example in intent['patterns']:
        tokens = nltk.word_tokenize(example)
        tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]
        question.append(' '.join(tokens))
        tags.append(intent['tag'])
tfidf_vectorizer = TfidfVectorizer(stop_words='english')#
patterns = tfidf_vectorizer.fit_transform(question)
tags = np.array(tags)

# Train a machine
train_m = LogisticRegression(max_iter=1000)
train_m.fit(patterns, tags)

# function for processing and generating responses
def preprocess_input(text):
    # text= text.decode('utf-8')
    text = str(text)

    # text = "Your string"
    text_bytes = text.encode('utf-8')  # Convert string to bytes

# Use the bytes object with quote_from_bytes()
    # encoded_text = quote_from_bytes(text_bytes)

    tokens = nltk.word_tokenize(text)
    # tokens = nltk.TreebankWordTokenizer(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]
    return ''.join(tokens)



def predict_intent(text):
    X_test = tfidf_vectorizer.transform([text])
    y_pred = train_m.predict(X_test)
    return y_pred[0]

def generate_response(intent):
    for item in data['intents']:
        if item['tag'] == intent:
            responses = item['responses']
            return np.random.choice(responses)



def perform_google_search(userText):
    search_results = []
    query = userText
    for result in search(query, tld="co.in", num=5, stop=5, lang='en'):
        search_results.append(result)

    content = ''
    for url in search_results:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        for paragraph in paragraphs:
            content += paragraph.text.strip() + '\n'

    if content:
        return  content  ###"Here is some information related to your search:\n\n" +
    else:
        return "I'm sorry, I couldn't find any relevant information for your search."

def chatbot_response(userText):
    preprocessed_input = preprocess_input(userText)
    intent = predict_intent(preprocessed_input)

    if userText in question:
        intent = predict_intent(preprocessed_input)
        res = generate_response(intent)
        # print('response')
        return res
        # print(res)
    else:
        google_results = perform_google_search(userText)
        # print("WRGFETGTRSRHYRJHRE",google_results)
        if google_results:

            return "Here are some search results: \n  \t"  + "".join(google_results)
        
        else:
            return "I'm sorry, I couldn't find any information."
    
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

app.static_folder = 'static'

@app.route("/get", methods=['GET'])
def get():
    userText = request.args.get('msg')
    collection.insert_one({'usertext': userText}) 
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run(debug=True)

