import cv2
import pandas as pd
import os
import time

# We put the "Heavy" imports inside the script to avoid terminal freezing
print("Starting script... Please wait.")

def collect_data():
    # STEP 1: Get the name first!
    label_name = input("\n[STEP 1] Enter the sign name (e.g. HELLO): ").upper()
    
    print(f"\n[STEP 2] Loading AI and Camera for '{label_name}'... Please wait 10 seconds.")
    
    # Import here to avoid early crashes
    try:
        from mediapipe.python.solutions import hands as mp_hands
        from mediapipe.python.solutions import drawing_utils as mp_draw
    except ImportError:
        print("Error: MediaPipe not found. Run 'pip install mediapipe'")
        return

    # Initialize MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=1, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. If you have a built-in camera, it should be 0.")
        return

    data_list = []
    count = 0
    recording = False

    print(f"\n[STEP 3] Success! Look for the 'Capture Window'.")
    print("Instructions: Press 'S' to start recording, 'Q' to quit.")

    while count < 400:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if recording:
                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    row.append(label_name)
                    data_list.append(row)
                    count += 1

        # Display status on the webcam frame
        status = f"RECORDING: {count}/400" if recording else "READY - PRESS 'S' TO START"
        cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Capture Window", frame)

        key = cv2.waitKey(1)
        # This will accept both lowercase 's' and uppercase 'S'
        if key == ord('s') or key == ord('S'): 
            recording = True
            print("Recording started...")
        if key == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

    if data_list:
        df = pd.DataFrame(data_list)
        df.to_csv(f"{label_name}.csv", index=False)
        print(f"\nSUCCESS: {label_name}.csv created with {len(data_list)} frames!")
    else:
        print("\nNo data was saved.")

# IMPORTANT: This line MUST be at the very bottom
if __name__ == "__main__":
    collect_data()