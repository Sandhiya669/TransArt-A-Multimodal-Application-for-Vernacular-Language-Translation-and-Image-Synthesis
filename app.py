import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import DiffusionPipeline
from PIL import Image
import google.generativeai as genai
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
import io

import os

st.set_page_config(page_title="TransArt", layout="wide")
#loads model nllb
#caching to avoid reloading
#runs on cpu to manage memory usage
@st.cache_resource
def load_translation_model():
    with st.spinner("Loading translation model... This may take a moment."):
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            device_map="cpu",              # Run on CPU to avoid meta device errors
            low_cpu_mem_usage=True
        )
        return tokenizer, model

#loads model sd
#checks gpu availability to optimize performance
#model loads with appropriate dtype based on device
@st.cache_resource
def load_image_generation_model():
    """Loads and caches the Stable Diffusion model pipeline."""
    with st.spinner("Loading image generation model (Stable Diffusion)... This will take time and RAM."):
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            # st.info("NVIDIA GPU (CUDA) detected. Using optimized settings.")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            # st.info("Running on CPU. Image generation will be very slow.")

        # Load the pre-trained pipeline
        pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch_dtype,
            use_safetensors=True
        )
        pipe = pipe.to(device)
    return pipe, device

#google gen.ai model with api key stored in environment variables
genai.configure(api_key=os.environ.get("GEMINI_API_KEY")
                
#generates content 
#handles exceptions to provide feedback in case of error                
def generate_creative_content(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Produces creative written content within 100 words based on this: '{prompt}'")
        return response.text
    except Exception as e:
        return f"Error generating creative content: {e}"
        
# Load both models when the app starts.
translation_tokenizer, translation_model = load_translation_model()
image_pipe, image_device = load_image_generation_model()

#transaltes text tamil to english
#tokenizes input,generates output 
def translate_text(tokenizer, model, tamil_text):
    tokenizer.src_lang = "tam_Taml"

    # Tokenize and translate
    inputs = tokenizer(tamil_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    output = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn")
    )

    # Decode and display
    english_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return english_text

#image genera tion using sd
#runs the model without tracking gradients
def generate_image_with_sd(pipe, device, prompt):
    enhanced_prompt = f"{prompt}"
    
    # Run the pipeline
    with torch.no_grad():
        image = pipe(prompt=enhanced_prompt, num_inference_steps=20).images[0]
    return image
#listens speech
#handles error related to speech
def recognize_tamil_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        with st.spinner("Listening... Speak in Tamil"):
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
        try:
            tamil_text = recognizer.recognize_google(audio, language="ta-IN")
            return tamil_text
        except sr.UnknownValueError:
            st.error("Could not understand speech.")
        except sr.RequestError:
            st.error("Request to Google failed.")
        return None
        

tamil_text=""
#section intializes the session state for tamil text & allows users to record audio
#process the audio to text
if "my_text" not in st.session_state:
    st.session_state.my_text = "ஒரு அழகான பூனை ஜன்னல் அருகே அமர்ந்திருக்கிறது"

audio_bytes = audio_recorder(text="Click to Record Tamil", pause_threshold=3.0)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio_data = recognizer.record(source)
        try:
            tamil_text = recognizer.recognize_google(audio_data, language="ta-IN")
            st.success("Tamil Text: " + tamil_text)
            st.session_state.my_text = tamil_text
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Google Speech API error: {e}")


tamil_input = st.text_area(
    "Enter Tamil text here:",
    value=st.session_state.my_text,
    height=100,
    help="Type or paste your descriptive Tamil text in this box."
)


if st.button("Translate and Generate Image"):
    if tamil_input:
        # st.success(f"Original (Tamil): {tamil_input}")

        with st.spinner("Translating with NLLB model..."):
            english_text = translate_text(translation_tokenizer, translation_model, tamil_input)
        st.success(f"English Translation: {english_text}")

        with st.spinner("Generating image with Stable Diffusion... (This can take several minutes on CPU)"):
            generated_image = generate_image_with_sd(image_pipe, image_device, english_text)
        
        if generated_image:
            st.image(generated_image, caption=english_text, use_container_width=True)

        with st.spinner("Generating creative content..."):
            creative_output = generate_creative_content(english_text)
        
        st.subheader("Creative Content from Gemini")
        st.text_area("Creative Output", creative_output, height=200)

    else:
        st.warning("Please enter some Tamil text before generating.")
else:
    st.info("Enter some Tamil text and click the button to start.")
