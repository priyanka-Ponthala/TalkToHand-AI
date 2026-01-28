import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
from gtts import gTTS
import pygame  # Used for reliable audio playback
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# 1. Initialize Pygame Mixer for audio
pygame.mixer.init()

def speak_text(text):
    """Function to convert text to speech and play it"""
    try:
        filename = "temp_voice.mp3"
        # Create audio file
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        
        # Play audio file
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        # Wait until finished or delete after use
        while pygame.mixer.music.get_busy():
            continue
            
        pygame.mixer.music.unload()
        os.remove(filename) # Clean up file
    except Exception as e:
        print(f"Audio Error: {e}")

# 2. Load the AI Model and Dictionary
print("Loading AI Model...")
model = tf.keras.models.load_model('sign_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

last_spoken = ""
frame_counter = 0

print("Translator LIVE! Using Google Voice Engine. Hold a sign to hear it.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            prediction = model.predict(np.array([landmarks]), verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)
            sign_text = label_encoder.inverse_transform([class_id])[0]

            if confidence > 0.9:
                cv2.putText(frame, f"Gesture: {sign_text}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if sign_text == last_spoken:
                    frame_counter += 1
                else:
                    frame_counter = 0
                    last_spoken = sign_text

                # Speak if steady for 35 frames
                if frame_counter == 35:
                    print(f"--- SPEAKING: {sign_text} ---")
                    speak_text(sign_text)
                    frame_counter = 0

    cv2.imshow("TalkToHand AI - gTTS Mode", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()