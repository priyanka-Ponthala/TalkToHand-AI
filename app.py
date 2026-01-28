import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
import speech_recognition as sr
from gtts import gTTS
import os
import time
import pygame # Used for playing the gTTS audio
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# --- INITIALIZE AUDIO ENGINE ---
if not pygame.mixer.get_init():
    pygame.mixer.init()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="TalkToHand AI", layout="wide", page_icon="🤟")
st.title("🤟 TalkToHand AI: Real-Time Bidirectional Translator")
st.markdown("---")

# --- LOAD AI MODEL & ENCODER ---
@st.cache_resource
def load_model_assets():
    model = tf.keras.models.load_model('sign_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_assets()

# --- HELPER FUNCTION: SPEAK TEXT ---
def speak_text(text):
    tts = gTTS(text=text, lang='en')
    filename = "temp_speech.mp3"
    tts.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    # Wait for audio to finish playing
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.music.unload()
    os.remove(filename)

# --- UI LAYOUT: TWO COLUMNS ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.header("📽️ Sign Language to Speech")
    run_translator = st.checkbox("Toggle Webcam Translator")
    frame_placeholder = st.empty()
    st.info("Instructions: Perform your sign and hold it steady for 1 second to hear the audio.")

with col2:
    st.header("👂 Hearing Person's Input")
    st.write("Click the button below to speak back to the Deaf user.")
    
    if st.button("🎤 Start Listening"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening for response...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = recognizer.listen(source, timeout=5)
                # 1. Convert speech to text
                raw_text = recognizer.recognize_google(audio).upper()
                st.success(f"Hearing Person said: **{raw_text}**")
                
                # 2. MATCH FILENAME LOGIC (Fixing the Space vs Underscore issue)
                search_filename = raw_text.replace(" ", "_")
                gif_path = f"assets/{search_filename}.gif"
                
                # 3. Display visual cue
                if os.path.exists(gif_path):
                    st.image(gif_path, caption=f"Visual Cue: {raw_text}", use_column_width=True)
                else:
                    st.warning(f"No visual cue found for '{raw_text}'.")
                    st.caption(f"Check if assets/{search_filename}.gif exists.")
            except Exception as e:
                st.error("I couldn't hear anything. Please try again.")

# --- LIVE TRANSLATOR LOGIC ---
if run_translator:
    cap = cv2.VideoCapture(0)
    # Use the Direct Import trick for MediaPipe stability on Windows
    hands_engine = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=1, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5
    )
    
    last_spoken = ""
    frame_counter = 0

    while run_translator:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access Webcam.")
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_engine.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks (x, y, z for all 21 points = 63 features)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # AI Prediction
                prediction = model.predict(np.array([landmarks]), verbose=0)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                sign_text = label_encoder.inverse_transform([class_id])[0]

                if confidence > 0.9:
                    # Draw detection text on the BGR frame for the placeholder
                    cv2.putText(frame, f"{sign_text} ({int(confidence*100)}%)", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Audio Counter Logic
                    if sign_text == last_spoken:
                        frame_counter += 1
                    else:
                        frame_counter = 0
                        last_spoken = sign_text

                    # Speak if held for 30 frames
                    if frame_counter == 30:
                        # Display the word in Streamlit text briefly
                        with col1:
                            st.write(f"📢 Speaking: **{sign_text}**")
                        speak_text(sign_text)
                        frame_counter = 0

        # Update the video frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")

    cap.release()