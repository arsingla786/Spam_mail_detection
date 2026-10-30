import streamlit as st
import pickle as pkl
import string 
import nltk 
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()

# ---------- NEW: page config (must be the first Streamlit command) ----------
st.set_page_config(
    page_title="SMS Spam Detector",
    layout="centered",
)

# ---------- NEW: minimal custom CSS ----------
st.markdown("""
<style>
    .stApp { background-color: #f7f8fa; }
    h1 {
        text-align: center;
        font-weight: 700;
        color: #111827 !important;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .stTextInput input {
        border-radius: 8px;
        padding: 12px;
    }
    div.stButton { text-align: center; }
    div.stButton > button {
        background-color: #111827;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #374151;
        color: #ffffff;
    }
    .result-card {
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 0.5rem;
        text-align: center;
        font-size: 1rem;
    }
    .spam-card { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .ham-card  { background-color: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

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


# ---------- NEW: sidebar ----------
with st.sidebar:
    st.subheader("About")
    st.write("Classifies an SMS as Spam or Not Spam using a TF-IDF vectorizer "
             "and a trained machine learning model.")
    st.divider()
    st.subheader("How it works")
    st.markdown("""
1. Lowercase and tokenize
2. Remove stopwords and symbols
3. Stem the words
4. TF-IDF vectorize
5. Predict
""")


st.title('SMS Spam Detector')
st.markdown('<p class="subtitle">Paste a message below to check whether it is spam.</p>',
            unsafe_allow_html=True)

input_sms = st.text_input("Enter the Message ")

# ---------- NEW: centered button layout ----------
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_clicked = st.button('Predict')

if predict_clicked:
#step 1 -> preprocess 

 with st.spinner('Analyzing...'):
  transformed_sms = transform_text(input_sms)

#step 2-> vectorize 
  vector_input = tfidf.transform([transformed_sms])

#step 3 -> predict 
  result = prediction = model.predict(vector_input)[0]

#step 4 display

 if(result == 1):
    st.header('Spam')
    st.markdown('<div class="result-card spam-card">This message looks like spam. Be careful with links and personal information.</div>',
                unsafe_allow_html=True)
 else:
    st.header('Not Spam')
    st.markdown('<div class="result-card ham-card">This message looks safe.</div>',
                unsafe_allow_html=True)

 # ---------- NEW: show how the text was processed ----------
 with st.expander("Processed text"):
    st.code(transformed_sms if transformed_sms else "(empty)")

# ---------- NEW: footer ----------
st.markdown('<div class="footer">Built with Streamlit</div>', unsafe_allow_html=True)