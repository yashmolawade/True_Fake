import streamlit as st
import pickle

@st.cache(allow_output_mutation=True)
def load_model():
    try:
        with open('NB_model_cv_n.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

def predict_news(news_text):
    prediction = model.predict([news_text])
    return prediction[0]

def main():
    st.title('News Authenticity Prediction')

    news_text = st.text_area('Enter the news text:', '')

    if st.button('Predict'):
        if len(news_text.strip()) == 0:
            st.warning("Please enter some news text!")
        else:
            prediction = predict_news(news_text)
            if prediction == 1:
                st.write('The news is: True')
            else:
                st.write('The news is: Fake')

if __name__ == '__main__':
    main()
