# Custom CSS to style the page
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('https://www.transparenttextures.com/patterns/newspaper.png') fixed; /* Ensure the background image covers the entire container */
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
        cursor: pointer; /* Add cursor pointer on hover */
        transition: all 0.3s ease; /* Add smooth transition */
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Darker shade of blue on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)
