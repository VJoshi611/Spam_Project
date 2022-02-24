#importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline             # creating pipeline
from sklearn.naive_bayes import MultinomialNB    #MultinomialNB good for discrete data
from sklearn.feature_extraction.text import CountVectorizer  # for text to vectorization

# importing dataset using pandas
df=pd.read_csv("spam.csv")
df.head()

#converting category attributes to numeric for spam and ham
df.Category=df.Category.apply(lambda x:1 if x == "spam" else 0)
# 1 == spam 

#display dataset
df.head()

#splitting data into train and test 
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(df.Message,df.Category,test_size=0.25,random_state=0)

# no. of rows and columns in dataset
df.shape

#building pipeline for vectorizer and algorithm
clf=Pipeline([
      ('CountVectorizer',CountVectorizer()),
      ('nb',MultinomialNB())
    ])

# fitting algo for train and test
clf.fit(Xtrain,ytrain)

#predicting test data using trained model
clf.predict(Xtest,ytest)

# predicting user input for spam or not
clf.predict(['Hey mohan, can we get together to watch footbal game tomorrow?'])

# getting model accuracy
clf.score(Xtest,ytest)

# saving model using pickle
import pickle
pickle.dump(clf,open('spam.pkl',"wb"))