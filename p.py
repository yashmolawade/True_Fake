import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords as nltk_stopwords
import spacy

# Load machine learning model
with open('your_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load CountVectorizer
with open('countvectorizer.pkl', 'rb') as f:
    count_vectorizer = pickle.load(f)

# NLTK stopwords
nltk_stopwords_set = set(nltk_stopwords.words('english'))

# spaCy stopwords
spacy_stopwords_set = spacy.lang.en.stop_words.STOP_WORDS

# Combine both sets
combined_stopwords_set = nltk_stopwords_set | spacy_stopwords_set

# Tokenize and lemmatize text
nlp = spacy.load('en_core_web_sm')
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in combined_stopwords_set and token.text.isalpha()]
    return ' '.join(tokens)

# Streamlit UI
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
            st.write('This is likely a fake news.')
        else:
            st.write('This is likely a true news.')
    else:
        st.write('Please enter some text.')
