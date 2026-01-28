import speech_recognition as sr

def listen_and_convert():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[TalkToHand] Listening to the hearing person...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        print("Processing voice...")
        text = recognizer.recognize_google(audio)
        print(f"Result: {text}")
        return text
    except Exception as e:
        print("Sorry, I couldn't hear you clearly.")
        return ""

if __name__ == "__main__":
    while True:
        listen_and_convert()