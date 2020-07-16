# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 13:03:49 2020

@author: vbhoj
"""

#pip install flask
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
model = pickle.load(open('Spam_Classifier.pkl','rb'))


@app.route('/')
def home():
    return render_template('spam.html')
        # return "<center><h1>Welcome to homepage</h1></center>"


@app.route('/predict', methods=['POST','GET'])
def predict():
    sentence = request.form['spamclass']
    data = [sentence]
    # print(data,type(data),sentence,type(sentence))
    cv = CountVectorizer()
    vect = cv.transform(data).toarray()    
    output = spam_detect_model.predict(vect)
    # print(output,"outputttttttttttttttttt")
    if output is ():
        return "There is some error"
    else:
        if output == 0:
            return render_template('spam.html', prediction_text = 'This text does not have spam words')
        else:
            return render_template('spam.html', prediction_text = 'This text contains spam words')
# # app.run()

if __name__ == "__main__":
    # app.run(debug = True)
    app.run()
       
    
    
