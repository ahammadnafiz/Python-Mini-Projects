import os
import threading
import base64
import io
import logging
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from gtts import gTTS
import pygame
from speech_recognition import Microphone, Recognizer, UnknownValueError
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
import customtkinter as ctk

logging.basicConfig(level=logging.INFO)
os.environ["GOOGLE_API_KEY"] = "AIzaSyADTZNqgOittRnTXuZTh8wSn_yFI-73_1c"

class WebcamStream:
    def __init__(self, index=0):
        self.stream = cv2.VideoCapture(index)
        self.running = False
        self.lock = threading.Lock()
        self.frame = None

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.stream.read()
            if not ret:
                logging.error("Failed to capture frame from webcam.")
                self.stop()
                break
            with self.lock:
                self.frame = frame

    def read(self, encode=False):
        with self.lock:
            frame = self.frame.copy()
        if encode:
            _, buffer = cv2.imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return
        try:
            response = self.chain.invoke(
                {"prompt": prompt, "image_base64": image.decode()},
                config={"configurable": {"session_id": "unused"}}
            ).strip()
            logging.info("Response: %s", response)
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            if response:
                self._tts(response)
        except Exception as e:
            logging.error("Error processing request: %s", e)

    def _tts(self, response, slow=False):
        try:
            tts = gTTS(text=response, lang='en', slow=slow)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            pygame.mixer.init()
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            logging.info("Spoken Response: %s", response)
        except Exception as e:
            logging.error("TTS error: %s", e)

    def _create_inference_chain(self, model):
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

class AssistantApp:
    def __init__(self, root, assistant, webcam_stream):
        self.root = root
        self.assistant = assistant
        self.webcam_stream = webcam_stream
        self.recognizer = Recognizer()
        self.microphone = Microphone()
        self.voice_mode = False
        self.setup_gui()
        self.root.after(1, self.update_camera_feed)

    def setup_gui(self):
        self.root.title("AI Assistant App")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        main_frame = ctk.CTkFrame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        inner_frame = ctk.CTkFrame(main_frame, border_width=2, border_color="#9AD2CB")
        inner_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.canvas = ctk.CTkCanvas(inner_frame, width=440, height=480)
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Add a button to toggle voice mode
        self.voice_button = ctk.CTkButton(inner_frame, text="Start Voice Mode", command=self.toggle_voice_mode)
        self.voice_button.grid(row=2, column=0, pady=10)

    def toggle_voice_mode(self):
        if self.voice_mode:
            self.deactivate_voice_mode()
        else:
            self.activate_voice_mode()

    def activate_voice_mode(self):
        self.voice_mode = True
        self.voice_button.configure(text="Stop Voice Mode")  # Change 'config' to 'configure'
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        self.stop_listening = self.recognizer.listen_in_background(self.microphone, self.audio_callback)

    def deactivate_voice_mode(self):
        self.voice_mode = False
        self.voice_button.configure(text="Start Voice Mode")  # Change 'config' to 'configure'
        if hasattr(self, 'stop_listening'):
            self.stop_listening(wait_for_stop=False)


    def audio_callback(self, recognizer, audio):
        try:
            # Recognize speech from audio
            prompt = recognizer.recognize_google(audio)
            logging.info("Recognized speech: %s", prompt)

            # Read the latest frame from the webcam
            image = self.webcam_stream.read(encode=True)
            if image is None:
                logging.error("Failed to capture frame from webcam.")
                return

            # Process the prompt and image using the assistant
            self.assistant.answer(prompt, image)
        except UnknownValueError:
            logging.warning("Audio not recognized.")
        except Exception as e:
            logging.error("Error during audio recognition or processing: %s", e)


    def update_camera_feed(self):
        frame = self.webcam_stream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480), interpolation=cv2.INTER_AREA)
        photo_image = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        if not hasattr(self, 'camera_image_id'):
            self.camera_image_id = self.canvas.create_image(320, 240, image=photo_image, anchor=tk.CENTER)
        else:
            self.canvas.itemconfig(self.camera_image_id, image=photo_image)
        self.canvas.photo_image = photo_image  # Keep a reference to prevent garbage collection
        self.root.after(1, self.update_camera_feed)  # Update as fast as possible

def main():
    root = tk.Tk()
    webcam_stream = WebcamStream().start()
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    assistant = Assistant(model)
    app = AssistantApp(root, assistant, webcam_stream)
    root.mainloop()

if __name__ == "__main__":
    main()