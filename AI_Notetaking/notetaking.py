import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import base64
import logging
import datetime
from threading import Thread
import cv2
import customtkinter as ctk
import pytesseract
import speech_recognition as sr
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_google_genai import ChatGoogleGenerativeAIessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO)
os.environ["GOOGLE_API_KEY"] = "AIzaSyADTZNqgOittRnTXuZTh8wSn_yFI-73_1c"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path


class WebcamStream:
    def __init__(self, index=0):
        self.stream = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            logging.warning(f"Failed to open camera at index {index}, trying next index")
            self.stream = cv2.VideoCapture(1 - index, cv2.CAP_DSHOW)
        
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.running = False
        self.frame = None
        if not self.stream.isOpened():
            logging.error("Failed to open any camera")
        self.frame = self._create_default_frame()
    
    def _create_default_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera Feed", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.stream.read()
            if not ret:
                logging.error("Failed to capture frame from webcam.")
                self.stop()
                break
            self.frame = frame  # No need for lock or copy here

    def read(self, encode=False):
        frame = self.frame
        if frame is None:
            return None
        frame = cv2.flip(frame, 1)  # Mirror the image horizontally
        if encode:
            _, buffer = cv2.imencode(".jpg", frame)
            return base64.b64encode(buffer)
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

class NoteTakingApp:
    def __init__(self, root, model, webcam_stream):
        self.root = root
        self.model = model
        self.webcam_stream = webcam_stream
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        self.setup_gui()
        self.last_note_image = None
        self.inference_chain = self._create_inference_chain()
        self.notes_data = []
        self.root.after(1, self.update_camera_feed)
        
    def setup_gui(self):
        self.root.title("AI Note Taking App")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.canvas = ctk.CTkCanvas(self.root, width=640, height=480)
        self.canvas.pack(padx=10, pady=10)

        self.snap_button = ctk.CTkButton(self.root, text="Snap", command=self.capture_photo)
        self.snap_button.pack(pady=10)

        self.notes_frame = ctk.CTkFrame(self.root)
        self.notes_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.notes_text = ctk.CTkTextbox(self.notes_frame, height=200)
        self.notes_text.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.image_label = ctk.CTkLabel(self.notes_frame, text="", image=None)
        self.image_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.tags_entry = ctk.CTkEntry(self.root, placeholder_text="Enter tags (comma-separated)")
        self.tags_entry.pack(pady=10, fill=tk.X, padx=10)

        self.save_button = ctk.CTkButton(self.root, text="Save", command=self.save_note)
        self.save_button.pack(pady=10)
        
        self.record_button = ctk.CTkButton(self.root, text="Record Voice Note", command=self.record_voice_note)
        self.record_button.pack(pady=5)

        self.ocr_button = ctk.CTkButton(self.root, text="Extract Text (OCR)", command=self.extract_text_from_image)
        self.ocr_button.pack(pady=5)

        self.search_entry = ctk.CTkEntry(self.root, placeholder_text="Search notes...")
        self.search_entry.pack(pady=5, fill=tk.X, padx=10)
        self.search_entry.bind('<Return>', self.search_notes)

    def capture_photo(self):
        frame = self.webcam_stream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_base64 = self.webcam_stream.read(encode=True)
        self.last_note_image = Image.fromarray(frame_rgb)
        self.process_image(image_base64)

    def process_image(self, image_base64):
        prompt = '''
        You are a note-taking assistant equipped with image recognition capabilities.
        Analyze the image and provide:
        1. A concise, engaging title for the note (max 5 words)
        2. Description of the main subject (1 sentence)
        3. Additional details or context (1-2 sentences)
        4. Personal observations or reflections (1 sentence)
        5. A list of 3-5 relevant tags, comma-separated

        Structure your response as:
        Title: [Note Title]
        Note:
        [Your structured note here]
        Tags: [tag1], [tag2], [tag3], ...
        '''
        response = self.inference_chain.invoke(
            {"prompt": prompt, "image_base64": image_base64.decode()},
            config={"configurable": {"session_id": "unused"}}
        ).strip()
        
        title, note, tags = self.parse_ai_response(response)
        self.display_note(image_base64, title, note, tags)
    
    def parse_ai_response(self, response):
        lines = response.split('\n')
        title = ""
        note = []
        tags = []

        for line in lines:
            if line.startswith("Title:"):
                title = line.replace('Title:', '').strip()
            elif line.startswith("Tags:"):
                tags = [tag.strip() for tag in line.replace('Tags:', '').split(',')]
                break
            elif line and not line.startswith("Note:"):
                note.append(line.strip())

        return title, '\n'.join(note), tags if tags else []

    def display_note(self, image_base64, title, note, tags):
        self.notes_text.delete("1.0", tk.END)
        self.notes_text.insert(tk.END, f"{title}\n\n{note}\n\nTags: {', '.join(tags)}")
        self.display_image(image_base64)
        self.tags_entry.delete(0, tk.END)
        self.tags_entry.insert(0, ', '.join(tags))

    def display_image(self, image_base64):
        image_array = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
        photo_image = ctk.CTkImage(light_image=Image.fromarray(image), dark_image=Image.fromarray(image), size=(200, 200))
        self.image_label.configure(image=photo_image)
        self.image_label.image = photo_image

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

    def save_note(self):
        if self.last_note_image is None:
            logging.warning("No image to save.")
            return

        note = self.notes_text.get("1.0", tk.END).strip()
        title = note.split('\n')[0]
        tags = [tag.strip() for tag in self.tags_entry.get().split(",") if tag.strip()]

        timestamp = int(datetime.datetime.now().timestamp())
        image_filename = f"{timestamp}.jpg"
        note_filename = f"{timestamp}.txt"

        save_dir = "saved_notes"
        os.makedirs(save_dir, exist_ok=True)

        image_path = os.path.join(save_dir, image_filename)
        note_path = os.path.join(save_dir, note_filename)

        self.last_note_image.save(image_path)
        with open(note_path, "w") as f:
            f.write(note)

        self.notes_data.append({"title": title, "note": note, "tags": tags, "image": image_filename, "text": note_filename})
        logging.info(f"Saved note: {title}")
    
    def record_voice_note(self):
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                self.notes_text.insert(tk.END, f"\n\nVoice Note: {text}")
        except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError) as e:
            logging.error(f"Voice recognition error: {e}")
    
    def extract_text_from_image(self):
        if self.last_note_image is None:
            logging.warning("No image available for OCR.")
            return
        
        try:
            text = pytesseract.image_to_string(self.last_note_image)
            if text.strip():
                self.notes_text.insert(tk.END, f"\n\nExtracted Text:\n{text}")
            else:
                logging.info("No text found in the image.")
        except pytesseract.TesseractError as e:
            logging.error(f"OCR error: {e}")
    
    def search_notes(self, event=None):
        query = self.search_entry.get().lower()
        results = []
        for note in self.notes_data:
            if (query in note['title'].lower() or 
                query in note['note'].lower() or 
                any(query in tag.lower() for tag in note['tags'])):
                results.append(note)
        
        if results:
            result = results[0]  # Just using the first result for now
            with open(os.path.join("saved_notes", result['text']), 'r') as f:
                self.notes_text.delete("1.0", tk.END)
                self.notes_text.insert(tk.END, f.read())
            
            image = Image.open(os.path.join("saved_notes", result['image']))
            image = image.resize((200, 200), Image.LANCZOS)
            photo = ctk.CTkImage(light_image=image, dark_image=image, size=(200, 200))
            self.image_label.configure(image=photo)
            self.image_label.image = photo
        else:
            logging.info("No matching notes found.")

    def _create_inference_chain(self):
        SYSTEM_PROMPT = """
        You are a note-taking assistant equipped with image recognition capabilities.
        When capturing photos, aim to provide descriptive notes that capture the essence
        of the scene or object in the image.

        Structure your notes like this:
        1. Description of the main subject (1 sentence)
        2. Additional details or context (1-2 sentences)
        3. Personal observations or reflections (1 sentence)

        Keep your notes concise, descriptive, and engaging. Imagine you're writing a
        snapshot description for someone who can't see the image.
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
        chain = prompt_template | self.model | StrOutputParser()
        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history"
        )

def main():
    root = tk.Tk()
    webcam_stream = WebcamStream().start()
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    app = NoteTakingApp(root, model, webcam_stream)
    root.mainloop()

if __name__ == "__main__":
    main()