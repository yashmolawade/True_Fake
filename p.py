import streamlit as st
import joblib
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords as nltk_stopwords
import spacy

nlp = spacy.load('en_core_web_sm')

model = load('NB_model_cv_n.pkl')
count_vectorizer = load('cv_n.pkl')

nltk_stopwords_set = set(nltk_stopwords.words('english'))
spacy_stopwords_set = nlp.Defaults.stop_words
combined_stopwords_set = nltk_stopwords_set | spacy_stopwords_set

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in combined_stopwords_set and token.text.isalpha()]
    return ' '.join(tokens)

st.set_page_config(
    page_title="VerityGuard: Defending Truth",
    page_icon="🛡️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .reportview-container {
        background: url('https://www.transparenttextures.com/patterns/newspaper.png') fixed;
        background-size: cover;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        border-radius: 10px;
        border: 2px solid #6c757d;
        padding: 10px;
        font-size: 18px;
        color: #495057;
    }
    .stButton>button {
        background-color: #007bff;
        color: #ffffff;
        font-weight: bold;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('VerityGuard: Defending Truth')

st.write('Enter a news headline or text:')
text_input = st.text_area('Input Text', '')

if st.button('Analyze'):
    if text_input.strip() != '':
        processed_text = preprocess_text(text_input)
        vectorized_text = count_vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)
        if prediction[0] == 1:
            st.write('This is likely a True news.', unsafe_allow_html=True)
        else:
            st.write('This is likely a Fake news.', unsafe_allow_html=True)
    else:
        st.write('Please enter some text.', unsafe_allow_html=True)
