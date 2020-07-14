# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:37:26 2020

@author: vbhoj
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle

messages = pd.read_csv(r"C:\Users\vbhoj\Downloads\NLP\Spam Classfier\SpamCollection",
                  sep="\t" ,names=['label','message'])

# messages = pd.read_csv(r"C:\Users\vbhoj\Downloads\NLP\Spam Classfier\spam.csv")



stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# stemming
corpus = []

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]', ' ',messages['message'][i])
    # print(review)
    
    review  =review.lower()
    review = review.split()
    # print(review)
    review = [stemmer.stem(word) for word in review if word not in stopwords.words("english")]
    review = ' '.join(review)
    corpus.append(review)
    
    
    
# Bag of words    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = .20 , random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)


# prediction of the spam
y_pred = spam_detect_model.predict(X_test)


from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test,y_pred)

'Accuracy score'
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)


# saving model file into disk
pickle.dump(spam_detect_model, open('Spam_Classifier.pkl','wb'))



