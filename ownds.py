import speech_recognition as sr
from transformers import pipeline

# Initialize the recognizer
recognizer = sr.Recognizer()


def load_dataset(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def voice_to_text():
    # Capture voice input
    with sr.Microphone() as source:
        print("Please ask your question:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        # Convert speech to text
        question = recognizer.recognize_google(audio)
        print("You asked: " + question)
        return question
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    return None


def get_answer(question, context):
    # Use a pretrained model for Q&A
    qa_pipeline = pipeline("question-answering")
    answer = qa_pipeline(question=question, context=context)
    return answer['answer']


if __name__ == "__main__":
    # Load your document or dataset
    dataset_path = 'C:/Users/dhars/PycharmProjects/NLP/best_buy_laptops_2024.csv'  # Change this to the path of your dataset in DOC format
    context = load_dataset(dataset_path)

    # Get question from voice input
    question = voice_to_text()

    if question:
        # Get the answer from the dataset
        answer = get_answer(question, context)
        print("Answer: " + answer)