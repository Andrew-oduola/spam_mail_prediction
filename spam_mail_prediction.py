import numpy as np
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

# Load the model and vectorizer
@st.cache_resource
def load_model():
    with open('spam_mail_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_vectorizer():
    with open('mail_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

# Predict spam or not
def spam_prediction(input_data):
    model = load_model()
    vectorizer = load_vectorizer()
    input_data_transformed = vectorizer.transform([input_data])
    prediction = model.predict(input_data_transformed)
    return prediction[0]

# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Spam Mail Prediction", page_icon="📧", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stTextArea>div>div>textarea {
                font-size: 16px;
            }
            .stMarkdown {
                font-size: 18px;
            }
            .created-by {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("📧 Spam Mail Prediction")
    st.markdown("This app predicts whether an email is spam or not based on its content.")

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="created-by">Created by Andrew O.A.</p>', unsafe_allow_html=True)
        
        # Load and display profile picture
        try:
            profile_pic = Image.open("prof.jpeg")  # Replace with your image file path
            st.image(profile_pic, caption="Andrew O.A.", use_container_width=True, output_format="JPEG")
        except:
            st.warning("Profile image not found.")

        st.title("About")
        st.info("This app uses a machine learning model to predict whether an email is spam or not.")
        st.markdown("[GitHub](https://github.com/Andrew-oduola) | [LinkedIn](https://linkedin.com/in/andrew-oduola-django-developer)")

    result_placeholder = st.empty()

    # Input field for email content
    email_content = st.text_area("Email Content", height=200, value="I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times", help="Enter the content of the email")

    # Prediction button
    if st.button("Predict"):
        try:
            if email_content.strip() == "":
                st.warning("Please enter some email content.")
            else:
                prediction = spam_prediction(email_content)
                
                if prediction == 1:
                    prediction_text = "This email is classified as **Spam**."
                    result_placeholder.error(prediction_text)
                    st.error(prediction_text)
                else:
                    prediction_text = "This email is classified as **Not Spam**."
                    result_placeholder.success(prediction_text)
                    st.success(prediction_text)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            result_placeholder.error("An error occurred during prediction. Please check the input data.")

if __name__ == "__main__":
    main()