pip install joblib
import streamlit as st
import joblib

# Load the pre-trained model
def load_model():
    try:
        with open("pretrained_model.pkl", "rb") as f:
            model = joblib.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

def predict_news(news_text):
    if model is None:
        return None
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
            if prediction is None:
                st.error("Model could not be loaded!")
            elif prediction == 1:
                st.write('The news is: True')
            else:
                st.write('The news is: Fake')

if __name__ == '__main__':
    main()
