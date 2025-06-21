#using logistic regression to detect spam emails 
#spam = 1 , not spam = 0 


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

df  = pd.read_csv('C:\\email_spam_dataset_2000.csv')
print(df.head())


#data preprocessing
print(df.shape)
df.dropna(inplace=True)

#features and target
X = df['text']  #text is th mail content 

y = df['label']  #label - > spam=1 , not spam =0

#convert text message to numeric data 
tfidf = TfidfVectorizer(
    stop_words='english',   # remove common useless words
    max_df=0.9,              # ignore overly frequent terms (in >90% emails)
    min_df=3,                # ignore rare terms (in <3 emails)
    max_features=3000        # limit total number of features
)
X_vectorized = tfidf.fit_transform(X) 
#define train and test data

X_train,X_test,y_train,y_test  =train_test_split(X_vectorized,y,test_size=0.2,random_state=42,stratify=y)

#train the model  
logistic_model =  LogisticRegression() 
logistic_model.fit(X_train,y_train)

#predict the input mail is spam or not 
y_pred = logistic_model.predict(X_test)

#evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Plots and graphs 
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True , fmt='d',cmap='Blues')
plt.title('Confusion matrix for smap email detection model')
plt.show()
#testing the model 
def predict_email(text):
    text_msg = tfidf.transform([text])
    prediction = logistic_model.predict(text_msg)
    return "SPAM" if prediction[0] == 1 else "NOT SPAM"



print(predict_email('Congrats! you won a lottery. Click here to claim award.'))

print(predict_email('HI, the class will be at 8 pm tomorrow'))



