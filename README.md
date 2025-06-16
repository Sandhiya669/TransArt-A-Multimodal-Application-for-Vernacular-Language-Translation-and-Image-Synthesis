
# TransArt: A Multimodal Application for Vernacular
Language Translation and Image Synthesis

web-based application that first translates text from Tamil to English and then
uses the translated text to generate relevant images. This application aims to demonstrate the
seamless integration of language translation and creative AI to produce visual content from
textual descriptions.


## Installation

1.Install the required packages:

pip install streamlit
pip install torch
pip install transformers
pip install diffusers
pip install pillow
pip install google-generativeai
pip install SpeechRecognition
pip install audio-recorder-streamlit

2.Set up Google API key:

Get a Google Gemini API key
Replace the API key in the code with your own

## Usage/Examples

1.Run the Streamlit app:

streamlit run app.py

2.Use the application:

This application allows users to interact with Tamil language content seamlessly. To record Tamil speech, simply click the "Click to Record Tamil" button, which captures your audio input. For text input, type your Tamil text in the designated text area. Once you have your content ready, click "Translate and Generate Image" to translate the Tamil text into English and generate an AI image based on the translation, enabling you to create unique and creative content using Gemini AI.
