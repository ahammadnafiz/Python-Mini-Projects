import os
import requests
import json
import io
from dotenv import load_dotenv
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import base64
import logging
from threading import Thread
import ctypes
import speech_recognition as sr
from threading import Event
import cv2
import customtkinter as ctk
import mimetypes
import pytesseract
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO)
load_dotenv('.env')
os.environ["GOOGLE_API_KEY"] = os.getenv('API_KEY')
NOTION_TOKEN = os.getenv('NOTION_TOKEN')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  

class WebcamStream:
    def __init__(self, index=0):
        self.stream = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            logging.warning(f"Failed to open camera at index {index}, trying next index")
            self.stream = cv2.VideoCapture(1 - index, cv2.CAP_DSHOW)
        
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 520)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 440)

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

        self.setup_gui()
        self.last_note_image = None
        self.inference_chain = self._create_inference_chain()
        self.notes_data = []
        self.root.after(1, self.update_camera_feed)
        
    def setup_gui(self):
        self.root.title("Synapto")
        ctk.set_appearance_mode("#070707")
        ctk.set_default_color_theme("blue")
        # self.root.geometry('520x440')

        # Create a main frame to hold the UI elements
        main_frame = ctk.CTkFrame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Create a notebook (tabs)
        notebook = ctk.CTkTabview(main_frame, width=510, height=640)
        notebook.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Create the first tab for the existing UI
        main_tab = notebook.add("Synapto")
        main_tab.grid_columnconfigure(0, weight=1)
        main_tab.grid_rowconfigure(0, weight=1)

        # Create the second tab for saved notes
        notes_tab = notebook.add("Saved Notes")
        notes_tab.grid_columnconfigure(0, weight=1)
        notes_tab.grid_rowconfigure(0, weight=1)

        # Add the existing UI to the first tab
        inner_frame = ctk.CTkFrame(main_tab, border_width=2, border_color="#9AD2CB", width=520, height=440)
        inner_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Create a canvas for camera feed
        self.canvas = ctk.CTkCanvas(inner_frame, width=520, height=440)
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Create a snap button with an icon
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        camera_light_path = os.path.join(assets_dir, "camera_light.png")
        camera_dark_path = os.path.join(assets_dir, "camera_dark.png")

        light_image = Image.open(camera_light_path)
        dark_image = Image.open(camera_dark_path)

        snap_icon = ctk.CTkImage(light_image=light_image, dark_image=dark_image, size=(20, 20))
        self.snap_button = ctk.CTkButton(inner_frame, text="Snap", image=snap_icon, compound="left",
                                        command=self.capture_photo, fg_color="#e76f51", hover_color="#f4a261")
        self.snap_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        voice_light_path = os.path.join(assets_dir, "voice_light.png")

        light_image = Image.open(voice_light_path)

        voice_icon = ctk.CTkImage(light_image=light_image, size=(20, 20))
        self.voice_button = ctk.CTkButton(inner_frame, text="Voice Note", image=voice_icon, compound="left",
                                        command=self.toggle_voice_recording, fg_color="#e76f51", hover_color="#f4a261")
        self.voice_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")

        # Create a frame for notes and image display
        notes_frame = ctk.CTkFrame(inner_frame)
        notes_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        notes_frame.grid_rowconfigure(0, weight=1)
        notes_frame.grid_columnconfigure(0, weight=1)
        notes_frame.grid_columnconfigure(1, weight=0)

        self.notes_text = ctk.CTkTextbox(notes_frame, height=130, font=("Helvetica", 12))
        self.notes_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.image_label = ctk.CTkLabel(notes_frame, text="", image=None)
        self.image_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.tags_entry = ctk.CTkEntry(inner_frame, placeholder_text="Enter tags (comma-separated)", font=("Helvetica", 12))
        self.tags_entry.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        separator = ctk.CTkLabel(inner_frame, text="-" * 50, font=("Helvetica", 10))
        separator.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Move the search entry to the notes tab
        search_frame = ctk.CTkFrame(notes_tab, width=520, height=440)
        search_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        search_frame.grid_columnconfigure(0, weight=1)

        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="Search notes...", font=("Helvetica", 12))
        self.search_entry.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.search_entry.bind('<Return>', self.search_notes)

        upload_light_path = os.path.join(assets_dir, "upload_light.png")
        upload_dark_path = os.path.join(assets_dir, "upload_dark.png")

        light_image = Image.open(upload_light_path)
        dark_image = Image.open(upload_dark_path)

        upload_icon = ctk.CTkImage(light_image=light_image, dark_image=dark_image, size=(20, 20))
        self.upload_button = ctk.CTkButton(inner_frame, text="Upload Image", image=upload_icon, compound="left",
                                        command=self.upload_image, fg_color="#e76f51", hover_color="#f4a261")
        self.upload_button.grid(row=5, column=0, padx=10, pady=10, sticky="w")

        ocr_light_path = os.path.join(assets_dir, "ocr_light.png")
        ocr_dark_path = os.path.join(assets_dir, "ocr_dark.png")

        light_image = Image.open(ocr_light_path)
        dark_image = Image.open(ocr_dark_path)

        ocr_icon = ctk.CTkImage(light_image=light_image, dark_image=dark_image, size=(20, 20))
        self.ocr_button = ctk.CTkButton(inner_frame, text="Extract Text (OCR)", image=ocr_icon, compound="left",
                                        command=self.extract_text_from_image, fg_color="#e76f51", hover_color="#f4a261")
        self.ocr_button.grid(row=5, column=1, padx=10, pady=10, sticky="e")

        self.save_button = ctk.CTkButton(inner_frame, text="Save", fg_color="#e76f51", hover_color="#f4a261",
                                        command=self.save_note)
        self.save_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

        # Create a frame for the saved notes feed
        notes_feed_frame = ctk.CTkFrame(notes_tab, border_width=2, border_color="#9AD2CB", width=500, height=640)
        notes_feed_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        notes_feed_frame.grid_rowconfigure(0, weight=1)                         
        notes_feed_frame.grid_columnconfigure(0, weight=1)

        # Create a canvas to display the saved notes feed
        self.notes_canvas = ctk.CTkCanvas(notes_feed_frame,width=470, height=600)
        self.notes_canvas.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Create a vertical scrollbar for the notes canvas
        scrollbar = ctk.CTkScrollbar(notes_feed_frame, orientation="vertical", command=self.notes_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.notes_canvas.configure(yscrollcommand=scrollbar.set)

        # Create a frame to hold the notes feed
        self.notes_feed_frame = ctk.CTkFrame(self.notes_canvas, width=500, height=640)
        self.notes_canvas.create_window((0, 0), window=self.notes_feed_frame, anchor="nw")

        # Bind the canvas configure event to update the scrollregion
        self.notes_canvas.bind("<Configure>", lambda e: self.notes_canvas.configure(scrollregion=self.notes_canvas.bbox("all")))        
        
        # Load and display the saved notes
        self.load_saved_notes()
    
    def toggle_voice_recording(self):
        if not hasattr(self, 'voice_recording_thread'):
            self.start_voice_recording()
        else:
            self.stop_voice_recording()

    def start_voice_recording(self):
        self.voice_recording_event = Event()
        self.voice_recording_thread = Thread(target=self.record_voice, args=(self.voice_recording_event,))
        self.voice_recording_thread.start()

    def stop_voice_recording(self):
        if hasattr(self, 'voice_recording_event'):
            self.voice_recording_event.set()
            if hasattr(self, 'voice_recording_thread'):
                self.voice_recording_thread.join(timeout=1.0)  # Wait for 1 second for the thread to stop
                if self.voice_recording_thread.is_alive():
                    print("Warning: Voice recording thread did not stop gracefully. Terminating thread...")
                    try:
                        
                        thread_id = self.voice_recording_thread.ident
                        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
                        if res > 1:
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
                            print('Exception raise failure')
                    except Exception as e:
                        print(f"Error terminating voice recording thread: {e}")
                del self.voice_recording_thread
            del self.voice_recording_event

    def record_voice(self, stop_event):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Start speaking...")
            audio_data = recognizer.listen(source)
            print("Recognizing...")

            try:
                text = recognizer.recognize_google(audio_data)
                # Get the current cursor position in the notes_text widget
                cursor_pos = self.notes_text.index(tk.INSERT)
                self.notes_text.insert(cursor_pos, f"\n{text}")
            except sr.UnknownValueError:
                print("Speech recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Speech Recognition service; {e}")

        if not stop_event.is_set():
            self.record_voice(stop_event)
            
    def load_saved_notes(self):
        self.notes_data = []  # Clear the existing notes_data list
        save_dir = "saved_notes"

        # Check if the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create the directory if it doesn't exist

        # List all files in the saved_notes directory
        for filename in os.listdir(save_dir):
            if filename.endswith(".txt"):  # Assuming all notes are saved as .txt files
                note_filename = filename
                image_filename = filename.replace(".txt", ".jpg")  # Get corresponding image filename

                # Read the note content
                note_path = os.path.join(save_dir, note_filename)
                with open(note_path, 'r') as f:
                    note_content = f.read()

                # Extract title and tags from the note content
                lines = note_content.split('\n')
                title = lines[0]
                tags = lines[-1].split(": ")[1].split(", ")

                # Add the note data to the notes_data list
                self.notes_data.append({
                    "title": title,
                    "note": note_content,
                    "tags": tags,
                    "image": image_filename,
                    "text": note_filename
                })

        # Clear the existing notes feed
        for widget in self.notes_feed_frame.winfo_children():
            widget.destroy()

        if not self.notes_data:
            empty_label = ctk.CTkLabel(
                self.notes_feed_frame, 
                text="No saved notes available.", 
                font=("Helvetica", 16, "italic"), 
                fg_color="#e76f51", 
                anchor="center"
            )
            empty_label.pack(padx=10, pady=10, fill="both", expand=True)
            return

        for note in self.notes_data:
            note_frame = ctk.CTkFrame(self.notes_feed_frame, border_width=2, border_color="#9AD2CB", corner_radius=10)
            note_frame.pack(padx=10, pady=10, fill="x", expand=True)

            # Note frame internal packing
            note_frame_inner = ctk.CTkFrame(note_frame, fg_color=None)
            note_frame_inner.pack(padx=5, pady=5, fill="x", expand=True)

            # Load and display the note image
            image_path = os.path.join(save_dir, note["image"])
            try:
                pil_image = Image.open(image_path)
                pil_image = pil_image.resize((100, 100), Image.LANCZOS)
                photo_image = ctk.CTkImage(pil_image, size=(100, 100))
                image_label = ctk.CTkLabel(note_frame_inner, image=photo_image)
                image_label.image = photo_image  # Keep a reference to prevent garbage collection
                image_label.pack(side="left", padx=5, pady=5)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                image_label = ctk.CTkLabel(note_frame_inner, text="No Image", width=100, height=100, fg_color="#E0E0E0")
                image_label.pack(side="left", padx=5, pady=5)

            text_frame = ctk.CTkFrame(note_frame_inner, fg_color=None)
            text_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

            # Display the note title
            note_title = ctk.CTkLabel(text_frame, text=note["title"], font=("Helvetica", 14, "bold"), anchor="w")
            note_title.pack(fill="x")

            # Display the note text
            note_text = ctk.CTkLabel(text_frame, text=note["note"], wraplength=330, anchor="w", justify="left")
            note_text.pack(fill="x")

        # Update the canvas scrollregion
        self.notes_canvas.configure(scrollregion=self.notes_canvas.bbox("all"))
                
    def parse_ai_response(self, response):
        # Split the response into lines
        lines = response.strip().split('\n')

        # Extract title, note, and tags from the response
        title = None
        note = []
        tags = None

        for line in lines:
            if line.startswith("Title:"):
                title = line.split(":")[1].strip()
            elif line.startswith("Note:"):
                note.append(line.split(":")[1].strip())
            elif line.startswith("Tags:"):
                tags = line.split(":")[1].strip().split(", ")

        # Combine note lines into a single string
        note = '\n'.join(note)

        return title, note, tags
    
    def upload_image(self):
        file_path = tk.filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            with open(file_path, "rb") as f:
                image_base64 = base64.b64encode(f.read())
            self.process_image_with_ai(image_base64)

    def process_image_with_ai(self, image_base64):
        # Send the image to the AI model for processing
        prompt = '''
        You are a great note taker, for the uploaded images, generate notes contains import information.
        The images usually taken from slides, meeting notes, what you need to do is help to take notes from the images.
        Output the information directly, do not say anything else.
        Make it more like a note contains important information, not a description of the image.
        A good structure of generated notes is like: 
        <example-structure>
        ## <-title->
        <-a paragraph of summary->
        <-details information from the image->
        </example-structure>
        If there are some tables, try to extract all orignial information in table format.
        Use the same language as the image, do not change the language.
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

        # Convert image to RGB mode
        image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        image = image.convert("RGB")

        # Set the last_note_image attribute
        self.last_note_image = image

    def capture_photo(self):
        frame = self.webcam_stream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_note_image = Image.fromarray(frame_rgb)
        image_base64 = self.webcam_stream.read(encode=True)
        self.process_image(image_base64)

    def process_image(self, image_base64):
        prompt = '''
        You are a great note taker, for the uploaded images, generate notes contains import information.
        The images usually taken from slides, meeting notes, what you need to do is help to take notes from the images.
        Output the information directly, do not say anything else.
        Make it more like a note contains important information, not a description of the image.
        A good structure of generated notes is like: 
        <example-structure>
        ## <-title->
        <-a paragraph of summary->
        <-details information from the image->
        </example-structure>
        If there are some tables, try to extract all orignial information in table format.
        Use the same language as the image, do not change the language.

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
        image = cv2.flip(image, 1)  # Mirror the image horizontally for display
        image = cv2.resize(image, (130, 130), interpolation=cv2.INTER_AREA)
        pil_image = Image.fromarray(image)
        photo_image = ctk.CTkImage(pil_image, size=(130, 130))
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
    

    def create_notion_page(self, token, database_id, title, content, tags, image_path=None):
        url = "https://api.notion.com/v1/pages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }

        # Prepare the content blocks
        blocks = [
            {
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"type": "text", "text": {"content": title}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": content}}]
                }
            }
        ]

        # Add image block if image_path is provided
        if image_path:
            # Use a placeholder URL for the image
            placeholder_image_url = "https://via.placeholder.com/400x300.png?text=Image+Placeholder"
            blocks.append({
                "object": "block",
                "type": "image",
                "image": {
                    "type": "external",
                    "external": {
                        "url": placeholder_image_url
                    }
                }
            })

        data = {
            "parent": {"database_id": database_id},
            "properties": {
                "Name": {
                    "title": [{"text": {"content": title}}]
                },
                "Tags": {
                    "multi_select": [{"name": tag} for tag in tags]
                }
            },
            "children": blocks
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print("Page created successfully in Notion!")
        else:
            print(f"Failed to create page in Notion: {response.status_code} - {response.text}")
            print(f"Request data: {json.dumps(data, indent=2)}")  # For debugging
            
    def save_note(self):
        if self.last_note_image is None:
            logging.warning("No image to save.")
            return

        note = self.notes_text.get("1.0", tk.END).strip()
        title = note.split('\n')[0]
        tags = [tag.strip() for tag in self.tags_entry.get().split(",") if tag.strip()]
        
        # Save note locally
        image_filename = f"{title}.jpg"
        note_filename = f"{title}.txt"

        save_dir = "saved_notes"
        os.makedirs(save_dir, exist_ok=True)

        image_path = os.path.join(save_dir, image_filename)
        note_path = os.path.join(save_dir, note_filename)

        self.last_note_image.save(image_path)
        with open(note_path, "w") as f:
            f.write(note)

        self.notes_data.append({"title": title, "note": note, "tags": tags, "image": image_filename, "text": note_filename})
        logging.info(f"Saved note locally: {title}")
        
        # Save note to Notion
        notion_token = NOTION_TOKEN
        notion_database_id = '70ef93d9e15c46048efc2513bbe7b67e'
        self.create_notion_page(notion_token, notion_database_id, title, note, tags, image_path)
        
        self.load_saved_notes()
    
    def extract_text_from_image(self):
        if self.last_note_image is None:
            logging.warning("No image available for OCR.")
            return
        
        try:
            text = pytesseract.image_to_string(self.last_note_image.transpose(Image.FLIP_LEFT_RIGHT))  # Flip image for OCR
            if text.strip():
                self.notes_text.insert(tk.END, f"\n\nExtracted Text:\n{text}")
            else:
                logging.info("No text found in the image.")
        except pytesseract.TesseractError as e:
            logging.error(f"OCR error: {e}")
    
    def search_notes(self, event=None):
        # Switch to the "Saved Notes" tab
        # self.root.notebook.select(self.root.notebook.tabs()[1])

        query = self.search_entry.get().lower()
        save_dir = "saved_notes"
        results = []

        # List all files in the saved_notes directory
        for filename in os.listdir(save_dir):
            if filename.endswith(".txt"):  # Assuming all notes are saved as .txt files
                note_path = os.path.join(save_dir, filename)
                with open(note_path, 'r') as f:
                    note_content = f.read().lower()  # Read note content and convert to lowercase
                    if query in note_content:
                        # If query is found in note content, add the note to results
                        image_filename = filename.replace(".txt", ".jpg")  # Get corresponding image filename
                        results.append({"text": filename, "image": image_filename, "content": note_content})

        # Clear the existing notes feed
        for widget in self.notes_feed_frame.winfo_children():
            widget.destroy()

        for i, result in enumerate(results):
            note_frame = ctk.CTkFrame(self.notes_feed_frame, border_width=1, border_color="#9AD2CB")
            note_frame.grid(row=i, column=0, padx=5, pady=5, sticky="ew")
            note_frame.grid_columnconfigure(0, weight=1)
            note_frame.grid_columnconfigure(1, weight=0)

            # Load and display the note image
            image_path = os.path.join(save_dir, result["image"])
            pil_image = Image.open(image_path)
            pil_image = pil_image.resize((100, 100), Image.LANCZOS)
            photo_image = ctk.CTkImage(pil_image, size=(100, 100))
            image_label = ctk.CTkLabel(note_frame, image=photo_image)
            image_label.image = photo_image  # Keep a reference to prevent garbage collection
            image_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

            # Display the note text
            note_text = ctk.CTkLabel(note_frame, text=result["content"], wraplength=330, anchor="w", justify="left")
            note_text.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Update the canvas scrollregion
        self.notes_canvas.configure(scrollregion=self.notes_canvas.bbox("all"))

    def _create_inference_chain(self):
        SYSTEM_PROMPT = """
        You are a note-taking assistant equipped with image recognition capabilities.
        When capturing photos, aim to provide descriptive notes that capture the essence
        of the scene or object in the image.

        Structure your notes like this:
        1. Description of the main subject (2 sentence) Must give it, if it is code snippet give additional details
        2. Additional details or context (2-4 sentences)
        3. Personal observations or reflections (1 sentence)
        4. If there is text on the image please describe it

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

    def cleanup(self):
        if hasattr(self, 'voice_recording_thread'):
            self.stop_voice_recording()
        if hasattr(self, 'webcam_stream'):
            self.webcam_stream.stop()

def main():
    root = tk.Tk()
    webcam_stream = WebcamStream().start()
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    app = NoteTakingApp(root, model, webcam_stream)

    try:
        root.mainloop()
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()