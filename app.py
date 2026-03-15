import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
import speech_recognition as sr
from gtts import gTTS
import os
import base64
import time
import google.generativeai as genai

# --- FIX FOR MEDIAPIPE ATTRIBUTE ERROR ---
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# --- 1. CORE PAGE CONFIGURATION ---
st.set_page_config(page_title="TalkToHand AI", layout="wide", page_icon="🤟")

# --- 2. AI GRAMMAR ENGINE SETUP (GEMINI) ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    llm = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("API Key missing! Please add GEMINI_API_KEY to your .streamlit/secrets.toml")

def fix_grammar(keyword_list):
    if not keyword_list: return ""
    raw_input = " ".join([k.replace("_", " ") for k in keyword_list])
    prompt = f"Translate these sign language keywords into one natural, polite English sentence: {raw_input}. Return ONLY the corrected sentence."
    try:
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return raw_input

# --- 3. LOGIC HELPERS ---
def clean_for_audio(text):
    return text.replace("_", " ").strip()

def get_gif_label(text):
    text = text.upper().strip()
    f_triggers = ["F", "EFF", "IF", "OFF", "FOR", "AF", "EF", "S", "PH", "HALF", "FOUR", "FRY"]
    if any(word == text for word in f_triggers) or "LETTER F" in text:
        return "f"
    
    phonetic_map = {
        "A": "a", "HAY": "a", "AY": "a", "EYE": "a", "EH": "a", "AN": "a",
        "B": "b", "BEE": "b", "BE": "b", "ME": "b", "DEE": "b", "PEE": "b",
        "C": "c", "SEE": "c", "SEA": "c", "SHE": "c", "SI": "c", "SAY": "c",
        "THANK YOU": "thank_you", "I LOVE YOU": "i_love_you",
        "HELLO": "hello", "YES": "yes", "NO": "no", "HELP": "help", "PLEASE": "please"
    }
    if text in phonetic_map: return phonetic_map[text]
    if "LETTER" in text: return text.split()[-1].lower()[0]
    return text.lower().replace(" ", "_")

def display_gif(sign_label):
    path = f"assets/{sign_label}.gif"
    if os.path.exists(path):
        with open(path, "rb") as f:
            data_url = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" width="450" style="border-radius:15px; border:4px solid #4CAF50;">',
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Sign asset for '{sign_label}' not found.")

def speak(text):
    try:
        clean_text = clean_for_audio(text)
        tts = gTTS(text=clean_text, lang='en')
        tts.save("voice.mp3")
        st.audio("voice.mp3", format="audio/mp3", autoplay=True)
    except Exception as e:
        st.error(f"Audio Error: {e}")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('sign_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

model, label_encoder = load_assets()

# --- 4. SESSION STATE ---
if 'sentence' not in st.session_state:
    st.session_state.sentence = []

# --- 5. UI NAVIGATION ---
st.sidebar.title("🎮 TalkToHand AI")
page = st.sidebar.selectbox("Navigate:", ["🏠 Home", "📽️ Sign to Speech", "👂 Speech to Sign"])

if page == "🏠 Home":
    st.title("TalkToHand AI: Smart Translator 🤟")
    st.write("Welcome to the bidirectional translation bridge.")
    st.image("https://img.freepik.com/free-vector/sign-language-concept-illustration_114360-6340.jpg", width=500)

elif page == "📽️ Sign to Speech":
    st.title("Sign Language to Natural English 🔊")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        run_cam = st.toggle("Start Camera")
        vid_area = st.empty()
    with c2:
        st.subheader("Detected Phrases")
        word_list_display = st.empty()
        
        if st.button("✨ Translate & Speak (AI)"):
            if st.session_state.sentence:
                with st.spinner("AI is thinking..."):
                    proper_sentence = fix_grammar(st.session_state.sentence)
                    st.success(f"**AI Translation:** {proper_sentence}")
                    speak(proper_sentence)
        
        if st.button("🗑️ Clear All"):
            st.session_state.sentence = []
            word_list_display.info("List cleared.")

    if run_cam:
        cap = cv2.VideoCapture(0)
        # Using the explicitly imported mp_hands.Hands
        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands_detector:
            last_sign, frames_held = "", 0
            while run_cam:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_detector.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hl in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                        coords = []
                        for lm in hl.landmark: coords.extend([lm.x, lm.y, lm.z])
                        
                        out = model.predict(np.array([coords]), verbose=0)
                        label = label_encoder.inverse_transform([np.argmax(out)])[0]
                        conf = np.max(out)
                        
                        if conf > 0.96:
                            cv2.putText(frame, f"{label} {int(conf*100)}%", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            if label == last_sign: frames_held += 1
                            else: frames_held, last_sign = 0, label
                            
                            if frames_held == 25:
                                if label == "CLEAR": st.session_state.sentence = []
                                elif label != "SPACE":
                                    st.session_state.sentence.append(label)
                                    speak(label)
                                frames_held = 0
                                # Update UI sidebar/list
                                raw_words = " ".join([clean_for_audio(s) for s in st.session_state.sentence])
                                word_list_display.info(raw_words)
                
                vid_area.image(frame, channels="BGR")
            cap.release()

elif page == "👂 Speech to Sign":
    st.title("Hearing Person to Visual Sign 👂")
    text_input = st.text_input("Type a word:")
    if text_input:
        display_gif(get_gif_label(text_input))

    if st.button("🔴 Start Microphone"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            try:
                audio = r.listen(source, timeout=5)
                text = r.recognize_google(audio)
                st.write(f"Recognized: {text}")
                display_gif(get_gif_label(text))
            except:
                st.error("Speech Recognition failed.")