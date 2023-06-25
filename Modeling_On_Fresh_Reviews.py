import re
import numpy as np
import pandas as pd
import joblib
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import csv

datasets = pd.read_csv("D:\\Programming\\Python Files\\Sentimental Analysis\\Files\\a2_RestaurantReviews_FreshDump.tsv", delimiter="\t",quoting=3)
all_stopwords = stopwords.words("english")
all_stopwords.remove("not")

#Data Cleaning   <------------
corpus =[]
for i in range(0,100):
    review = re.sub('[^a-zA-Z]'," ",datasets["Review"][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in all_stopwords]
    review = " ".join(review)
    corpus.append(review)
    
#Data Transformation  <--------------
bow_path = "D:\\Programming\\Python Files\\Sentimental Analysis\\BOW_.pkl"
cv = pickle.load(open(bow_path,"rb"))
X_fresh = cv.transform(corpus).toarray()

#Calling Classifier   <-------------
cf_path = "D:\\Programming\\Python Files\\Sentimental Analysis\\Classifier Sentimental Model"
classifier = joblib.load(cf_path)
Y_pred = classifier.predict(X_fresh)

#Predictions         <-----------------
datasets["Predicted Label"] = Y_pred.tolist()
datasets.to_csv("D:\\Programming\\Python Files\\Sentimental Analysis\\Predicted_Sentimental.tsv",sep="\t",encoding='UTF-8',index=False)

df = pd.read_csv("D:\\Programming\\Python Files\\Sentimental Analysis\\Predicted_Sentimental.tsv", delimiter="\t", quoting=3)

# replace 0 with "negative" string and 1 with "positive" string
df["Predicted Label"] = df["Predicted Label"].replace({0: "negative", 1: "positive"})
df.to_csv("Predicted Sentiments.csv",index = False)

