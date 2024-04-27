from flask import Flask, request, render_template, url_for, redirect
import pickle
import joblib
import score
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__,template_folder='template')

train = pd.read_csv("/mnt/d/Applied ML/Assignment 4/data/train.csv")
test = pd.read_csv("/mnt/d/Applied ML/Assignment 4/data/test.csv")
val = pd.read_csv("/mnt/d/Applied ML/Assignment 4/data/validation.csv")

model_name = "Logistic Regression"

model = joblib.load("/mnt/d/Applied ML/Assignment 4//Best_LR.pkl")


#splitting the datframe into X and y
y_train, X_train = train["y_train"], train["X_train"]
y_val, X_val = val["y_val"], val["X_val"]
y_test, X_test = test["y_test"], test["X_test"]

tfidf = TfidfVectorizer()
train_tfidf = tfidf.fit_transform(X_train)




threshold=0.5

@app.route('/') 
def home():
    return render_template('spam.html')


@app.route('/spam', methods=['POST'])
def spam():
    sent = request.form['sent']
    label,prop=score.score(sent,model,threshold)
    lbl="Spam" if label == 1 else "Not spam"
    ans1 = f"""The input text is {sent}"""
    ans2 = f"""The prediction is {lbl}""" 
    ans3 = f"""The propensity score is {prop}"""
    return render_template('result.html', ans1 = ans1, ans2 = ans2, ans3 = ans3)


if __name__ == '__main__': 
    app.run(debug=True)