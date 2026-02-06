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
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# --- 1. SETUP ---
st.set_page_config(page_title="TalkToHand AI", layout="wide")
if not pygame.mixer.get_init():
    pygame.mixer.init()

# --- 2. THE ULTIMATE SPEECH-TO-SIGN FIX ---
def get_best_gif_match(raw_speech):
    # Standardize input
    text = raw_speech.upper().replace(".", "").strip()
    
    # 1. KEYWORD SEARCH (Most stable for phrases)
    # If the user says "Thank you very much", we find "THANK" and show THANK_YOU.gif
    if "THANK" in text: return "THANK_YOU"
    if "LOVE" in text: return "I_LOVE_YOU"
    if "PLEASE" in text: return "PLEASE"
    if "HELP" in text: return "HELP"
    if "HELLO" in text: return "HELLO"
    if "YES" in text: return "YES"
    if "NO" in text: return "NO"

    # 2. PHONETIC SEARCH (For single letters A, B, C, F)
    # Google often hears these words instead of letters
    phonetic_map = {
        "A": "A", "AY": "A", "EH": "A", "HEY": "A", "HAY": "A",
        "B": "B", "BEE": "B", "BE": "B", "PEE": "B",
        "C": "C", "SEE": "C", "SEA": "C", "SHE": "C",
        "F": "F", "EFF": "F", "IF": "F", "OFF": "F", "HALF": "F"
    }
    
    # Check words one by one
    words = text.split()
    for word in words:
        if word in phonetic_map:
            return phonetic_map[word]
            
    # 3. SPELLED OUT SEARCH (If it hears "T H A N K")
    joined = "".join(words)
    if "THANK" in joined: return "THANK_YOU"
    
    return text.replace(" ", "_")

# --- 3. UI HELPERS ---
def display_gif(file_path):
    with open(file_path, "rb") as f:
        data_url = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" width="400" style="border:5px solid #00FF00; border-radius:15px;">', unsafe_allow_html=True)

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("temp.mp3")
        pygame.mixer.music.load("temp.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): continue
        pygame.mixer.music.unload()
        os.remove("temp.mp3")
    except: pass

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('sign_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_assets()

# --- 4. THE APP ---
st.sidebar.title("TalkToHand AI")
page = st.sidebar.radio("Mode:", ["🏠 Home", "📽️ Sign-to-Speech", "👂 Speech-to-Sign"])

if page == "🏠 Home":
    st.title("TalkToHand AI 🤟")
    st.write("Bidirectional Translation System")

elif page == "📽️ Sign-to-Speech":
    st.header("Sign Language Translator")
    if 's_list' not in st.session_state: st.session_state.s_list = []
    
    run = st.checkbox("Webcam On")
    frame_place = st.empty()
    st.write(f"**Current Sentence:** {' '.join(st.session_state.s_list)}")

    if run:
        cap = cv2.VideoCapture(0)
        hands = mp_hands.Hands(min_detection_confidence=0.7)
        while run:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                    lm = []
                    for i in hl.landmark: lm.extend([i.x, i.y, i.z])
                    pred = model.predict(np.array([lm]), verbose=0)
                    if np.max(pred) > 0.9:
                        txt = encoder.inverse_transform([np.argmax(pred)])[0]
                        cv2.putText(frame, txt, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        # Add to list logic... (keep as is)
            frame_place.image(frame, channels="BGR")
        cap.release()

elif page == "👂 Speech-to-Sign":
    st.header("Hearing Person Input")
    st.write("Speak into the mic. I will try to find the sign.")
    
    if st.button("🎤 Start Listening"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.warning("Listening... (Speak clearly)")
            try:
                audio = r.listen(source, timeout=5)
                heard = r.recognize_google(audio)
                
                # --- DEBUGGING DISPLAY ---
                st.write(f"DEBUG: I heard raw text: `{heard}`")
                
                match = get_best_gif_match(heard)
                st.info(f"DEBUG: Matching with file: `{match}.gif`")
                
                path = f"assets/{match}.gif"
                if os.path.exists(path):
                    st.success(f"Showing Sign for: {match}")
                    display_gif(path)
                else:
                    st.error(f"File not found: {path}")
            except Exception as e:
                st.error(f"Error: {e}")