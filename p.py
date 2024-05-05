import streamlit as st
import joblib
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords as nltk_stopwords
import spacy

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load machine learning model
model = load('NB_model_cv_n.pkl')

# Load CountVectorizer
count_vectorizer = load('cv_n.pkl')

# NLTK stopwords
nltk_stopwords_set = set(nltk_stopwords.words('english'))

# spaCy stopwords
spacy_stopwords_set = nlp.Defaults.stop_words

# Combine both sets
combined_stopwords_set = nltk_stopwords_set | spacy_stopwords_set

# Tokenize and lemmatize text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in combined_stopwords_set and token.text.isalpha()]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# Custom CSS to style the page
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('https://www.transparenttextures.com/patterns/newspaper.png');
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
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Fake News Detection')

st.write('Enter a news headline or text:')
text_input = st.text_area('Input Text', '')

if st.button('Predict'):
    if text_input.strip() != '':
        # Preprocess text
        processed_text = preprocess_text(text_input)
        
        # Vectorize text
        vectorized_text = count_vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(vectorized_text)
        
        # Display prediction
        if prediction[0] == 1:
            st.write('This is likely a True news.')
        else:
            st.write('This is likely a Fake news.')
    else:
        st.write('Please enter some text.')
