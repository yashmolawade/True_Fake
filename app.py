import streamlit as st
import pickle

# Load the pre-trained model
with open("D:\\NLP Project\\Deployment\\NB_model_cv_n.pkl",'rb') as f:
    model = pickle.load(f)

def predict_text(text):
    prediction = model.predict([text])
    return prediction[0]

def main():
    st.title('Text Prediction')

    uploaded_file = st.file_uploader("Upload a Pickle file", type="pkl")
    if uploaded_file is not None:
        with open(uploaded_file.name, 'wb') as f:
            f.write(uploaded_file.getvalue())

    text = st.text_area('Enter text:', '')
    if st.button('Predict'):
        if uploaded_file is not None:
            with open(uploaded_file.name, 'rb') as f:
                loaded_model = pickle.load(f)
            prediction = predict_text(text)
            st.write('The text is:', prediction)
        else:
            st.write('Please upload a Pickle file')

if __name__ == '__main__':
    main()
