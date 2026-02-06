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

# --- 1. CORE PAGE CONFIGURATION ---
st.set_page_config(page_title="TalkToHand AI", layout="wide", page_icon="🤟")

if not pygame.mixer.get_init():
    pygame.mixer.init()

# --- 2. THE FIXERS (Audio & Logic) ---

def clean_for_audio(text):
    """Fixes the 'I_LOVE_YOU' spelling problem by removing underscores for natural speech"""
    return text.replace("_", " ").strip()

def get_gif_label(text):
    """Aggressive Phonetic Map to catch the difficult 'F' sound and others"""
    text = text.upper().strip()
    
    # 1. THE F-SHIELD: If any of these are heard, force it to 'f'
    f_triggers = ["F", "EFF", "IF", "OFF", "FOR", "AF", "EF", "S", "PH", "HALF", "FOUR", "FRY"]
    if any(word == text for word in f_triggers) or "LETTER F" in text:
        return "f"
    
    # 2. General Phonetic Map for A, B, C and Phrases
    phonetic_map = {
        # A Variants
        "A": "a", "HAY": "a", "AY": "a", "EYE": "a", "EH": "a", "AN": "a", "AND": "a",
        # B Variants
        "B": "b", "BEE": "b", "BE": "b", "ME": "b", "DEE": "b", "PEE": "b", "TEA": "b",
        # C Variants
        "C": "c", "SEE": "c", "SEA": "c", "SHE": "c", "SI": "c", "SAY": "c",
        # Word Phrases
        "THANK YOU": "thank_you", "I LOVE YOU": "i_love_you",
        "HELLO": "hello", "YES": "yes", "NO": "no", "HELP": "help", "PLEASE": "please"
    }
    
    if text in phonetic_map:
        return phonetic_map[text]
    
    # 3. Logic: If user says "Letter [X]", extract the last character
    if "LETTER" in text:
        parts = text.split()
        return parts[-1].lower()[0]

    # 4. Default: Lowercase and underscores
    return text.lower().replace(" ", "_")

# --- 3. HELPER FUNCTIONS ---

def display_gif(sign_label):
    """Looks for lowercase gifs in assets folder"""
    path = f"assets/{sign_label}.gif"
    if os.path.exists(path):
        with open(path, "rb") as f:
            data_url = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" width="450" style="border-radius:15px; border:4px solid #4CAF50;">',
            unsafe_allow_html=True
        )
    else:
        st.warning(f"File not found: {path}. Check your assets folder!")

def speak(text):
    """Speaks text naturally (sentence mode)"""
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

# --- 4. UI NAVIGATION ---
page = st.sidebar.selectbox("Select Mode:", ["🏠 Home", "📽️ Sign to Speech", "👂 Speech to Sign"])

if page == "🏠 Home":
    st.title("TalkToHand AI: Final Bridge 🤟")
    st.image("https://img.freepik.com/free-vector/sign-language-concept-illustration_114360-6340.jpg", width=500)
    st.write("Bidirectional Sign Language translation is ready.")

# --- PAGE 1: SIGN TO SPEECH ---
elif page == "📽️ Sign to Speech":
    st.title("Sign to Natural Speech 🔊")
    if 'sentence' not in st.session_state: st.session_state.sentence = []
    
    c1, c2 = st.columns([2, 1])
    with c1:
        run_cam = st.checkbox("Webcam Toggle")
        vid_area = st.empty()
    with c2:
        st.subheader("Recognized Phrase")
        readable_text = " ".join([clean_for_audio(s) for s in st.session_state.sentence])
        st.success(readable_text if readable_text else "Sign something to begin...")
        if st.button("🔊 Speak Full Thought"): speak(readable_text)
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
                        if label == last_sign: frames_held += 1
                        else: frames_held, last_sign = 0, label
                        if frames_held == 22:
                            if label == "CLEAR": st.session_state.sentence = []
                            else:
                                st.session_state.sentence.append(label)
                                speak(label)
                            frames_held = 0
                            st.rerun()
            vid_area.image(frame, channels="BGR")
        cap.release()

# --- PAGE 2: SPEECH TO SIGN (F-SOUND OPTIMIZED) ---
elif page == "👂 Speech to Sign":
    st.title("Speech to Visual Sign 👂")
    st.info("💡 **Tip:** If 'F' is not working, try saying **'Letter F'** or **'Eff'** clearly.")

    if st.button("🔴 Start Microphone"):
        r = sr.Recognizer()
        # Lower threshold (250) helps hear the quiet 'F' sound
        r.energy_threshold = 250 
        r.dynamic_energy_threshold = False
        
        with sr.Microphone() as source:
            st.info("Calibrating... please stay silent.")
            r.adjust_for_ambient_noise(source, duration=0.6)
            st.success("SPEAK NOW!")
            try:
                audio_data = r.listen(source, timeout=5, phrase_time_limit=3)
                heard_text = r.recognize_google(audio_data).upper()
                
                # Debug line so you can see what the AI thinks it heard
                st.write(f"AI Heard: `{heard_text}`")
                
                final_sign_name = get_gif_label(heard_text)
                display_gif(final_sign_name)
                
            except Exception as e:
                st.error("Speech Recognition failed. Try speaking closer to the mic or saying 'Letter F'.")