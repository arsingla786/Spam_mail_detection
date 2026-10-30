import streamlit as st
import pickle as pkl
import string 
import nltk 
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()

tfidf = pkl.load(open('vectorizer.pkl','rb'))
model = pkl.load(open('model.pkl','rb'))

def transform_text(text):
  #to lower case
  text = text.lower()
  #word tokenize
  text = nltk.word_tokenize(text)
  #remove special chars
  y = []
  for i in text:
    if i.isalnum():
      y.append(i)
  #remove stopwords and punctuation(/ , . ; etc)
  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  #stemming
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)


st.title('SMS Spam Detector')

input_sms = st.text_input("Enter the Message ")
if st.button('Predict'):
#step 1 -> preprocess 

 transformed_sms = transform_text(input_sms)

#step 2-> vectorize 
 vector_input = tfidf.transform([transformed_sms])

#step 3 -> predict 
 result = prediction = model.predict(vector_input)[0]

#step 4 display

 if(result == 1):
    st.header('Spam')
 else:
    st.header('Not Spam')

