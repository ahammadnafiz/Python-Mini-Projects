import base64
import logging
import os
from threading import Lock, Thread

import cv2
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv
from cv2 import VideoCapture, imencode
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

# Basic logging configuration
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv('.env')
os.environ["GOOGLE_API_KEY"] = os.getenv('API_KEY')

# Initialize the text-to-speech engine
engine = pyttsx3.init()


class WebcamStream:
    """Class to manage webcam streaming."""
    def __init__(self, index=0):
        self.stream = VideoCapture(index)
        self.running = False
        self.lock = Lock()
        self.frame = None

    def start(self):
        """Start streaming."""
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        """Update the frame."""
        while self.running:
            ret, frame = self.stream.read()
            if not ret:
                logging.error("Failed to capture frame from webcam.")
                self.stop()
                break
            with self.lock:
                self.frame = frame

    def read(self, encode=False):
        """Read the frame."""
        with self.lock:
            frame = self.frame.copy()
        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        return frame

    def stop(self):
        """Stop streaming."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()


class Assistant:
    """Class to manage the assistant."""
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        """Provide an answer based on the prompt and image."""
        if not prompt:
            return
        try:
            response = self.chain.invoke(
                {"prompt": prompt, "image_base64": image.decode()},
                config={"configurable": {"session_id": "unused"}}
            ).strip()
            logging.info("Response: %s", response)
            if response:
                self._tts(response)
        except Exception as e:
            logging.error("Error processing request: %s", e)

    def _tts(self, response, slow=False):
        """
        Convert text to speech using pyttsx3.

        Args:
            response (str): Text to be spoken.
            slow (bool): Whether to speak slowly.

        Returns:
            None
        """
        engine = pyttsx3.init()

        if slow:
            rate = 100  
        else:
            rate = 150  

        engine.setProperty('rate', rate)

        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)

        engine.say(response)
        engine.runAndWait()

    def _create_inference_chain(self, model):
        """Create an inference chain."""
        SYSTEM_PROMPT = """
        You are a witty, insightful assistant that can see and understand images. 
        When answering questions, aim to provide responses that are a bit more 
        substantial—about 2 to 4 sentences long. This allows you to offer more 
        context, share interesting observations, or add a touch of personality.

        Structure your responses like this:
        1. Direct Answer (1 sentence)
        2. Image Insight or Context (1-2 sentences)
        3. Personal Touch or Follow-up (1 sentence)

        For example:
        "The weather looks chilly. I see you're wearing a thick coat and scarf, 
        and the trees in the background are bare—classic winter scene. Don't forget 
        your gloves next time!"

        Keep your tone friendly, slightly witty, and conversational. Imagine you're 
        a perceptive friend who's good at reading situations. Feel free to make 
        educated guesses about what you see, but phrase them as observations, not 
        facts.
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}},
                    ]
                )
            ]
        )
        chain = prompt_template | model | StrOutputParser()
        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history"
        )


def audio_callback(recognizer, audio, assistant, webcam_stream):
    """Callback function for audio recognition."""
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, webcam_stream.read(encode=True))
    except sr.UnknownValueError:
        logging.warning("Audio not recognized.")
    except Exception as e:
        logging.error("Error during audio recognition: %s", e)


def process_multimodel_assistant():
    """Run the application."""
    webcam_stream = WebcamStream().start()
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    assistant = Assistant(model)

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    stop_listening = recognizer.listen_in_background(
        microphone,
        lambda recognizer, audio: audio_callback(recognizer, audio, assistant, webcam_stream)
    )

    try:
        while True:
            cv2.imshow("Webcam", webcam_stream.read())
            if cv2.waitKey(1) in [27, ord("q")]:
                break
    finally:
        webcam_stream.stop()
        cv2.destroyAllWindows()
        stop_listening(wait_for_stop=False)
