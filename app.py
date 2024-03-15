import os
import requests
import streamlit as st
from textblob import TextBlob
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import nltk
import ssl
import openai

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    raw_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in paragraphs])
    return text

def analyze_sentiment(text):
    blob = TextBlob(text)
    positive_sentences = []
    negative_sentences = []
    positive_percentage = 0
    negative_percentage = 0

    for sentence in blob.sentences:
        sentiment = sentence.sentiment.polarity
        if sentiment > 0:
            positive_sentences.append(sentence.raw)
        elif sentiment < 0:
            negative_sentences.append(sentence.raw)

    total_sentences = len(positive_sentences) + len(negative_sentences)
    if total_sentences > 0:
        positive_percentage = round((len(positive_sentences) / total_sentences) * 100, 2)
        negative_percentage = round((len(negative_sentences) / total_sentences) * 100, 2)

    return positive_percentage, negative_percentage

def main():
    # Download the necessary corpus data for TextBlob
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('brown')
        nltk.download('wordnet')
        nltk.download('movie_reviews')
    except Exception as e:
        st.error(f"Failed to download necessary data for TextBlob: {e}")

    # Header
   
    # Check if the user wants to write a text, upload a PDF file, or enter a URL
    option = st.radio("Select Input Type", ("URL", "Text", "PDF",))

    # Create area for the user to write the text
    if option == "Text":
        user_input = st.text_area("Enter Text", "")

        # Submit Button
        if st.button("Submit") and user_input != "":
            # Analyze sentiment
            positive_percentage, negative_percentage = analyze_sentiment(user_input)
            st.subheader("Sentiment Analysis")
            st.write(f"Positive Percentage: {positive_percentage}%")
            st.write(f"Negative Percentage: {negative_percentage}%")
        else:
            st.error("Please enter some text.")
    elif option == "URL":
        url_input = st.text_input("Enter URL", "")

        if st.button("Submit") and url_input != "":
            # Extract text from the URL
            text = extract_text_from_url(url_input)
            
            # Analyze sentiment
            positive_percentage, negative_percentage = analyze_sentiment(text)
            st.subheader("Sentiment Analysis")
            st.write(f"Positive Percentage: {positive_percentage}%")
            st.write(f"Negative Percentage: {negative_percentage}%")
           
        else:
            st.error("Please enter a URL.")
    else:
        # Create a file uploader for the user to upload the PDF file
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        # Creating a submit button for the PDF file
        if st.button("Submit") and uploaded_file is not None:
            # Extract text from a PDF file
            text = extract_text_from_pdf(uploaded_file)
            
            # Analyze sentiment
            positive_percentage, negative_percentage = analyze_sentiment(text)
            st.subheader("Sentiment Analysis")
            st.write(f"Positive Percentage: {positive_percentage}%")
            st.write(f"Negative Percentage: {negative_percentage}%")
           
        else:
            st.error("Please upload a PDF file.")

if __name__ == "__main__":
    main()
