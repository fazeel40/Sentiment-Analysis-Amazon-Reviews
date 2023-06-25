#Importing Libraries     <----------
import numpy as np
import pandas as pd
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import pickle

dataset = pd.read_csv("D:\\Programming\\Python Files\\Sentimental Analysis\\Files\\a1_RestaurantReviews_HistoricDump.tsv")

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

#Data Cleaning           <----------
corpus = []
for i in range(0,900):
    review = re.sub('[^a-zA-Z]'," ",dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = " ".join(review)
    corpus.append(review)
    
#Data Transformation    <----------
cv = CountVectorizer(max_features=1420)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#Saving Bow Dictionary for late use
bow_path = "D:\\Programming\\Python Files\\Sentimental Analysis\\BOW_.pkl"
pickle.dump(cv,open(bow_path,'wb'))

#Training and Test Sets
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.20,random_state=0)

#Model Fitting(Naive Bayes)
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

#Exporting Classifier File
clf_path = "D:\\Programming\\Python Files\\Sentimental Analysis\\Classifier Sentimental Model"
joblib.dump(classifier, clf_path)


#Mode_Performance
y_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test,y_pred)
print(cm)
print(accuracy_score(Y_test,y_pred))