import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load trained model and vectorizer
with open("sentiment_model1.pkl", "rb") as best_model_file:
    best_model = pickle.load(best_model_file)

with open("vectorizer1.pkl", "rb") as best_vec_file:
    best_vectorizer = pickle.load(best_vec_file)

# Load LabelEncoder
with open("label_encoder.pkl", "rb") as label_file:
    label_encoder = pickle.load(label_file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Text Preprocessing Function"""
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

import base64

# Function to encode image in Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load and encode background image
bg_image_path = "Image.jpg"  # Ensure this file exists in the directory
bg_image_base64 = get_base64_image(bg_image_path)

# Inject background image into CSS
# Inject background image into CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image_base64}"); /* Correct way */
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stTextInput > div > div > input {{
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 5px;
        font-size: 16px;
    }}
    .stButton > button {{
        background-color: #ff4b4b;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
        width: 100%;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Centered Title
st.markdown("<h1 style='text-align: center; color: white;'>Sentiment Analysis Web App</h1>", unsafe_allow_html=True)

# UI Elements inside a styled container
with st.container():
    # st.markdown('<div class="custom-box">', unsafe_allow_html=True)
    st.markdown("<h3>Enter a sentence to analyze its sentiment.</h3>", unsafe_allow_html=True)
    # Inject CSS to make the input text color black
    st.markdown(
        """
        <style>
        .stTextInput > div > div > input {
            color: black !important;
            background-color: white !important; /* White background */
            font-size: 16px; /* Optional: Adjust font size */
            }
        </style>
        """,
    unsafe_allow_html=True
    )

    # User Input Box
    user_input = st.text_input("Enter text:")

    if st.button("Analyze Sentiment"):
        if user_input:
            cleaned_text = preprocess_text(user_input)  # Preprocess input
            vectorized_text = best_vectorizer.transform([cleaned_text])  # Transform using saved vectorizer
            prediction = best_model.predict(vectorized_text)  # Predict

            # Reverse encoding to get the original label
            predicted_sentiment = label_encoder.inverse_transform(prediction)

            st.markdown(f"<h3 style='color: black;'><b>Predicted Sentiment:</b> {predicted_sentiment[0]}</h3>", unsafe_allow_html=True)

        else:
            st.warning("⚠️ Please enter some text.")
    st.markdown('</div>', unsafe_allow_html=True)

# Sentiment Data for Chart
sentiments = {
    "neutral": 8617,
    "worry": 8452,
    "happiness": 5194,
    "sadness": 5160,
    "love": 3801,
    "surprise": 2187,
    "fun": 1776,
    "relief": 1524,
    "hate": 1323,
    "empty": 827,
    "enthusiasm": 759,
    "boredom": 179,
    "anger": 110
}

# Convert data to lists
sentiment_labels = list(sentiments.keys())
sentiment_values = list(sentiments.values())


# Create Figure for Sentiment Distribution
fig, ax = plt.subplots(figsize=(10, 6))

# Set background transparency
fig.patch.set_alpha(0.3)  # Set figure background transparency
ax.patch.set_alpha(0.3)   # Set axes background transparency

# Plot Sentiment Bar Chart
sns.barplot(x=sentiment_labels, y=sentiment_values, ax=ax, palette="coolwarm")

# Customize Plot
ax.set_title("Sentiment Analysis", fontsize=16, fontweight="bold")
ax.set_xlabel("Sentiments", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45)

# Display Bar Chart in Streamlit
st.pyplot(fig)
