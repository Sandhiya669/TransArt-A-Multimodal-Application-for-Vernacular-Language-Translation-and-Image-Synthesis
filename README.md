
# TransArt: A Multimodal Application for Vernacular
Language Translation and Image Synthesis

To develop a web-based application that first translates text from Tamil to English and then
uses the translated text to generate relevant images. This application aims to demonstrate the
seamless integration of language translation and creative AI to produce visual content from
textual descriptions.


## Installation

1.Install the required packages:

pip install streamlit torch transformers diffusers pillow google-generativeai speechrecognition audio-recorder-streamlit

2.Set up Google API key:

Get a Google Gemini API key
Replace the API key in the code with your own

## Usage/Examples

1.Run the Streamlit app:

streamlit run app.py

2.Use the application:

Audio Input: Click "Click to Record Tamil" to record Tamil speech
Text Input: Type Tamil text in the text area
Generate: Click "Translate and Generate Image" to:

Translate Tamil to English
Generate an AI image based on the translation
Create creative content using Gemini AI

