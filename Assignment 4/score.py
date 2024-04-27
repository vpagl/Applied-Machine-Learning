import pandas as pd
import os
import joblib
import numpy as np
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))

train = pd.read_csv("/mnt/d/Applied ML/Assignment 4/data/train.csv")
val = pd.read_csv("/mnt/d/Applied ML/Assignment 4/data/validation.csv")
test = pd.read_csv("/mnt/d/Applied ML/Assignment 4/data/test.csv")

# Splitting the dataframe into X and y
y_train, X_train = train["y_train"], train["X_train"]
y_val, X_val = val["y_val"], val["X_val"]
y_test, X_test = test["y_test"], test["X_test"]

tfidf = TfidfVectorizer()
train_tfidf = tfidf.fit_transform(X_train)

def split_into_tokens(data):
    tokenized_words = []
    regex = r"\w+"
    
    for i in range(len(data)):
        tokenized_words.append(re.findall(regex, data[i]))
        
    return tokenized_words

def lemmatize(data):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    
    for i in range(len(data)):
        for word in data[i]:
            if word.lower() in stop_words:
                continue
            elif word in string.punctuation:
                continue
            else:
                lemmatized_words.append(lemmatizer.lemmatize(word).lower())
                
    return lemmatized_words

def score(text:str, model, threshold:float) -> tuple:
    token_words = split_into_tokens([text])
    text_lemmatized = lemmatize(token_words)
    text_processed = " ".join(text_lemmatized)
    prediction = float(model.predict_proba(tfidf.transform([text_processed]))[0][0]) <= threshold

    if prediction == 1:
        propensity = 1 - model.predict_proba(tfidf.transform([text_processed]))[0]
    else:
        propensity = model.predict_proba(tfidf.transform([text_processed]))[0]
    
    return prediction, float(propensity[0])
