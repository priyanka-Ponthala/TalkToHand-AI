import cv2
import numpy as np
import tensorflow as tf
import pickle
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# 1. Load the AI and the Dictionary
print("Loading AI Model...")
model = tf.keras.models.load_model('sign_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# 2. Setup Camera and MediaPipe
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

print("AI Translator is LIVE! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks for AI
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # 3. Predict!
            prediction = model.predict(np.array([landmarks]), verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Get the word (e.g., "HELLO")
            sign_text = label_encoder.inverse_transform([class_id])[0]

            # 4. Show result if AI is > 80% confident
            if confidence > 0.8:
                cv2.putText(frame, f"{sign_text} ({int(confidence*100)}%)", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("TalkToHand AI - Real Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()