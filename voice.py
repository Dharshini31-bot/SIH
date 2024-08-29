import speech_recognition as sr

def voice_to_text():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as the source for input
    with sr.Microphone() as source:
        print("Please speak something...")

        # Adjust for ambient noise and record the audio
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        # Convert speech to text
        text = recognizer.recognize_google(audio)
        print("You said: " + text)

    except sr.UnknownValueError:
        # If speech is unintelligible
        print("Sorry, I could not understand the audio")

    except sr.RequestError as e:
        # If the API was unreachable or unresponsive
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

if __name__ == "__main__":
    voice_to_text()