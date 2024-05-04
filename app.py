import streamlit as st
import pickle

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    with open("pretrained_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

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
