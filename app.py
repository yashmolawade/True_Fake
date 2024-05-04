import streamlit as st
import joblib

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("NB_model_cv_n.pkl")

model = load_model()

def predict_news(news_text):
    prediction = model.predict([news_text])
    return prediction[0]

def main():
    st.title('News Authenticity Prediction')

    news_text = st.text_area('Paste your news text here:', '', height=200)
    
    if st.button('Predict'):
        if len(news_text.strip()) == 0:
            st.warning("Please enter some text!")
        else:
            prediction = predict_news(news_text)
            if prediction == 1:
                st.write('The news is: True')
            else:
                st.write('The news is: Fake')

if __name__ == '__main__':
    main()
