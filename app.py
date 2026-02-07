import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
import speech_recognition as sr
from gtts import gTTS
import os
import pygame
import base64
import time
import google.generativeai as genai
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# --- 1. CORE PAGE CONFIGURATION ---
st.set_page_config(page_title="TalkToHand AI", layout="wide", page_icon="🤟")

if not pygame.mixer.get_init():
    pygame.mixer.init()

# --- 2. AI GRAMMAR ENGINE SETUP (GEMINI) ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
llm = genai.GenerativeModel('gemini-2.5-flash')

def fix_grammar(keyword_list):
    """Turns Sign Language keywords into natural English sentences using Gemini AI"""
    if not keyword_list:
        return ""
    
    # Convert list to string and remove underscores: ["THANK_YOU", "I"] -> "THANK YOU I"
    raw_input = " ".join([k.replace("_", " ") for k in keyword_list])
    
    prompt = f"Translate these sign language keywords into one natural, polite English sentence: {raw_input}. Return ONLY the corrected sentence."
    
    try:
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return raw_input # Fallback to raw if API fails

# --- 3. THE LOGIC FIXERS (Audio & Speech Mapping) ---

def clean_for_audio(text):
    return text.replace("_", " ").strip()

def get_gif_label(text):
    """Phonetic Map to catch difficult sounds like 'F' and single letters"""
    text = text.upper().strip()
    
    # F-Trigger Protection
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

# --- 4. HELPER FUNCTIONS ---

def display_gif(sign_label):
    path = f"assets/{sign_label}.gif"
    if os.path.exists(path):
        with open(path, "rb") as f:
            data_url = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" width="450" style="border-radius:15px; border:4px solid #4CAF50; box-shadow: 0px 4px 15px rgba(0,255,0,0.3);">',
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Visual asset for '{sign_label}' not found.")

def speak(text):
    try:
        clean_text = clean_for_audio(text)
        tts = gTTS(text=clean_text, lang='en')
        tts.save("voice.mp3")
        pygame.mixer.music.load("voice.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): continue
        pygame.mixer.music.unload()
        os.remove("voice.mp3")
    except: pass

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('sign_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

model, label_encoder = load_assets()

# --- 5. UI NAVIGATION ---
st.sidebar.title("🎮 TalkToHand AI")
page = st.sidebar.selectbox("Navigate:", ["🏠 Home", "📽️ Sign to Speech", "👂 Speech to Sign"])

if page == "🏠 Home":
    st.title("TalkToHand AI: Smart Translator 🤟")
    st.image("https://img.freepik.com/free-vector/sign-language-concept-illustration_114360-6340.jpg", width=500)
    st.markdown("""
    ### Features:
    - **Sign to Speech:** Real-time gesture detection with **AI Grammar Correction**.
    - **Speech to Sign:** Voice-activated visual cues with phonetic mapping.
    """)

# --- PAGE 1: SIGN TO SPEECH (With Gemini AI) ---
elif page == "📽️ Sign to Speech":
    st.title("Sign Language to Natural English 🔊")
    if 'sentence' not in st.session_state: st.session_state.sentence = []
    
    c1, c2 = st.columns([2, 1])
    with c1:
        run_cam = st.checkbox("Toggle Webcam")
        vid_area = st.empty()
    with c2:
        st.subheader("Recognized Keywords")
        raw_words = " ".join([clean_for_audio(s) for s in st.session_state.sentence])
        st.info(raw_words if raw_words else "Waiting for gestures...")
        
        # --- GEMINI AI INTEGRATION ---
        if st.button("✨ Translate & Speak (AI)"):
            if st.session_state.sentence:
                with st.spinner("Gemini AI is forming a sentence..."):
                    proper_sentence = fix_grammar(st.session_state.sentence)
                    st.success(f"**AI Translation:** {proper_sentence}")
                    speak(proper_sentence)
            else:
                st.warning("Please sign a few words first.")

        if st.button("🗑️ Clear All"):
            st.session_state.sentence = []
            st.rerun()

    if run_cam:
        cap = cv2.VideoCapture(0)
        hands_mp = mp_hands.Hands(min_detection_confidence=0.85)
        last_sign, frames_held = "", 0

        while run_cam:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            results = hands_mp.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                    coords = []
                    for lm in hl.landmark: coords.extend([lm.x, lm.y, lm.z])
                    out = model.predict(np.array([coords]), verbose=0)
                    label = label_encoder.inverse_transform([np.argmax(out)])[0]
                    
                    if np.max(out) > 0.96:
                        cv2.putText(frame, label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                        
                        if label == last_sign: 
                            frames_held += 1
                        else: 
                            frames_held, last_sign = 0, label
                        
                        if frames_held == 22:
                            if label == "CLEAR": 
                                st.session_state.sentence = []
                                st.rerun()
                            else:
                                if label not in ["SPACE"]:
                                    st.session_state.sentence.append(label)
                                    speak(label)
                                frames_held = 0
                                st.rerun()
            vid_area.image(frame, channels="BGR")
        cap.release()

# --- PAGE 2: SPEECH TO SIGN ---
elif page == "👂 Speech to Sign":
    st.title("Hearing Person to Visual Sign 👂")
    
    if st.button("🔴 Start Microphone"):
        r = sr.Recognizer()
        r.energy_threshold = 300 
        with sr.Microphone() as source:
            st.info("Listening...")
            r.adjust_for_ambient_noise(source, duration=0.6)
            try:
                audio_data = r.listen(source, timeout=5)
                heard_text = r.recognize_google(audio_data).upper()
                st.write(f"Recognized: `{heard_text}`")
                final_sign_name = get_gif_label(heard_text)
                display_gif(final_sign_name)
            except:
                st.error("Speech Recognition failed. Try again.")