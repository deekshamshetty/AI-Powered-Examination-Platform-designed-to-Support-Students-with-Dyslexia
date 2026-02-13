import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io

from googletrans import Translator
from googletrans import LANGUAGES
from gtts import gTTS
import uuid
import os

import speech_recognition as sr
from pydub import AudioSegment

st.set_page_config(page_title="AI Toolkit", layout="wide")
st.title("🧠 AI-Powered Examination Support System")

# ------------------------------
# Load BLIP model (image captioning)
# ------------------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip()

# ======================================================
# 1️⃣ IMAGE CAPTIONING MODULE
# ======================================================

st.header("📸 Image Captioning")
uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, width=350)

    inputs = processor(img, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)
    st.success(f"📝 Caption: {caption}")

    if st.button("🔊 Speak Caption"):
        filename = f"{uuid.uuid4()}.mp3"
        tts = gTTS(text=caption, lang="en")
        tts.save(filename)
        st.audio(open(filename, "rb").read(), format="audio/mp3")
        os.remove(filename)

# ======================================================
# 2️⃣ LANGUAGE TRANSLATION MODULE
# ======================================================

st.header("🌍 Language Translator")

translator = Translator()
text_input = st.text_area("Enter text to translate")

languages = LANGUAGES.values()
dest_lang = st.selectbox("Select a language", list(languages))

if st.button("Translate"):
    translated = translator.translate(text_input, dest=dest_lang)
    st.success(f"Translated Text:\n\n{translated.text}")

# ======================================================
# 3️⃣ TEXT TO SPEECH MODULE
# ======================================================

st.header("🔊 Text to Speech")
tts_text = st.text_area("Enter text to convert to speech")

if st.button("Convert to Speech"):
    filename = f"{uuid.uuid4()}.mp3"
    tts = gTTS(text=tts_text, lang="en")
    tts.save(filename)
    st.audio(open(filename, "rb").read(), format="audio/mp3")
    os.remove(filename)

# ======================================================
# 4️⃣ SPEECH TO TEXT MODULE
# ======================================================

st.header("🎤 Speech to Text (Real-Time Microphone Input)")

audio_input = st.audio_input("Record your voice")

if audio_input:
    st.audio(audio_input)

    # Save the audio to a temporary WAV file
    wav_path = f"{uuid.uuid4()}.wav"
    with open(wav_path, "wb") as f:
        f.write(audio_input.getvalue())

    recognizer = sr.Recognizer()

    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        st.success(f"Recognized Text: {text}")
    except Exception as e:
        st.error(f"Error: {e}")

    os.remove(wav_path)
