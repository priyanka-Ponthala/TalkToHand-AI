# TalkToHand AI: Bridging the Communication Gap 👐💬

**TalkToHand AI** is a real-time, bidirectional communication platform designed to facilitate seamless interaction between Sign Language users and non-signers. By utilizing advanced Computer Vision and Deep Learning, the system translates hand gestures into text and speech, while simultaneously converting spoken language into visual cues for the deaf and hard-of-hearing community.

---

## 🌟 The Problem
Communication between the Deaf and Hard-of-Hearing (DHH) community and the hearing population often relies on human interpreters, who are not always available or affordable. This creates significant barriers in essential sectors like healthcare, education, and public services.

## 💡 Our Solution
TalkToHand AI provides an accessible, camera-based alternative:
- **Sign-to-Text/Speech:** Translates Sign Language gestures into audible and readable formats for non-signers.
- **Speech-to-Visuals:** Converts spoken words into animations or visual cues, enabling the Sign Language user to understand the response in real-time.

---

## 🛠 Technical Architecture
The system is built on a modular pipeline designed for low latency and high accuracy:

1. **Input Capture:** Real-time video stream processed via **OpenCV**.
2. **Hand Tracking:** **MediaPipe** extracts 21 specific 3D hand landmarks (coordinates), reducing computational load by focusing only on joint movement rather than raw image pixels.
3. **Classification:** A custom **Convolutional Neural Network (CNN)** analyzes the landmark sequences to identify specific signs.
4. **Translation Engine:** 
    - **gTTS (Google Text-to-Speech):** Converts recognized signs into audio.
    - **SpeechRecognition:** Processes vocal responses back into text/visuals.
5. **Interface:** A high-performance web dashboard built with **Streamlit**.

---

## 🚀 Key Features
- **Real-Time Translation:** Low-latency inference for natural conversation flow.
- **Landmark-Based Recognition:** Robust performance across different lighting conditions and backgrounds.
- **Bidirectional Support:** Full two-way communication loop.
- **Accessibility Focused:** Designed for deployment in hospitals, classrooms, and customer service centers.

---
