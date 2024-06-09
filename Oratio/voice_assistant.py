import os
from dotenv import load_dotenv
import logging
import tkinter as tk
import customtkinter as ctk
import speech_recognition as sr
import pyttsx3
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from threading import Thread

# Basic logging configuration
logging.basicConfig(level=logging.INFO)

# Set comtypes logging level to WARNING to avoid informational messages
logging.getLogger('comtypes').setLevel(logging.WARNING)

load_dotenv('.env')
os.environ["GOOGLE_API_KEY"] = os.getenv('API_KEY')

engine = pyttsx3.init()


class VoiceAssistant:
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
        self.groq_chat = ChatGroq(groq_api_key=self.groq_api_key, model_name='llama3-8b-8192')
        self.prompt = self._create_chat_prompt()
        self.conversation = LLMChain(
            llm=self.groq_chat,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory,
        )

    def answer(self, prompt):
        """
        Answer user's query using the conversation model.

        Args:
            prompt (str): User's query.

        Returns:
            None
        """
        if not prompt:
            return
        try:
            response = self.conversation.predict(human_input=prompt)
            print("Response:", response)
            self._tts(response)
        except Exception as e:
            print("Error processing request:", e)

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

    def _create_chat_prompt(self):
        """
        Create a chat prompt template.

        Returns:
            ChatPromptTemplate: The chat prompt template.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=
                              '''
                            You are a witty, insightful assistant that can Analyze and provide insights across diverse topics. 
                            When answering questions, aim to provide responses that are a bit more 
                            substantial—about 1 to 3 sentences long. This allows you to offer more 
                            context, share interesting observations, or add a touch of personality.
                            Keep your tone friendly, slightly witty, and conversational. Imagine you're 
                            a perceptive friend who's good at listening situations. Feel free to make 
                            educated guesses about what you listen, but phrase them as observations, not 
                            facts.
                              '''),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )
        return prompt


def audio_callback(recognizer, audio, assistant):
    """
    Callback function for audio recognition.

    Args:
        recognizer (Recognizer): Speech recognizer instance.
        audio (AudioData): Audio data to be recognized.
        assistant (VoiceAssistant): Voice assistant instance.

    Returns:
        None
    """
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt)
    except sr.UnknownValueError:
        print("Audio not recognized.")
    except Exception as e:
        print("Error during audio recognition:", e)


def handle_voice_assistant():
    """
    Handles voice input and passes it to the voice assistant.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    audio_callback(recognizer, audio, voice_assistant)


def run_async(func):
    """
    Decorator to run a function asynchronously.

    Args:
        func: Function to be executed.

    Returns:
        wrapper: Decorated function.
    """
    def wrapper(*args, **kwargs):
        thread = Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread
    return wrapper


@run_async
def process_voice_assistant():
    """
    Process voice assistant asynchronously.
    """
    handle_voice_assistant()


GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize the assistants
groq_api_key = GROQ_API_KEY  
voice_assistant = VoiceAssistant(groq_api_key)