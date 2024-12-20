import os
import requests
import json
import io
import base64
import logging
import ctypes
from threading import Thread, Event

import numpy as np
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import colorchooser
from tkinter import font as tkfont

import cv2
import pytesseract
import speech_recognition as sr
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI


# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from a .env file
load_dotenv('.env')

# Set environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv('API_KEY')
NOTION_TOKEN = os.getenv('NOTION_TOKEN')

# Set the path for the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

        self.setup_gui()
        self.setup_editor_tab()
        self.last_note_image = None
        self.inference_chain = self._create_inference_chain()
        self.notes_data = []
        self.root.after(1, self.update_camera_feed)
        
    def setup_gui(self):
        self.root.title("Synapto - Smart Note-Taking Assistant")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.root.geometry('1200x800')
        self.root.minsize(1000, 700)

        # Define color scheme for dark theme
        self.colors = {
            "primary": "#1e1e1e",  # Dark background
            "secondary": "#252526",  # Slightly lighter background
            "accent": "#007acc",  # Bright blue for highlights
            "text": "#d4d4d4",  # Light grey for text
            "button": "#3a3a3a"  # Dark grey for buttons
        }

        # Create a main frame to hold the UI elements
        main_frame = ctk.CTkFrame(self.root, fg_color=self.colors["primary"])
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Create a notebook (tabs)
        self.notebook = ctk.CTkTabview(main_frame, fg_color=self.colors["primary"])
        self.notebook.pack(fill="both", expand=True)

        # Create the tabs
        main_tab = self.notebook.add("Capture")
        self.notebook.add("Editor")
        notes_tab = self.notebook.add("Saved Notes")
        
        # Setup Capture tab
        self.setup_capture_tab(main_tab)

        # Setup Saved Notes tab
        self.setup_notes_tab(notes_tab)

    def setup_capture_tab(self, tab):
        # Left frame for camera feed and buttons
        left_frame = ctk.CTkFrame(tab, fg_color=self.colors["primary"])
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Camera feed
        self.camera_frame = ctk.CTkFrame(left_frame, corner_radius=10, fg_color=self.colors["secondary"])
        self.camera_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Adjust canvas size to match webcam stream size
        self.canvas = ctk.CTkCanvas(self.camera_frame, width=640, height=480, highlightthickness=0, bg=self.colors["secondary"])
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)

        # Buttons frame
        buttons_frame = ctk.CTkFrame(left_frame, fg_color=self.colors["primary"])
        buttons_frame.pack(fill="x", padx=10, pady=(0, 10))

        # Load button icons
        assets_dir = "assets"  
        camera_icon = ctk.CTkImage(Image.open(f"{assets_dir}/camera_light.png"), size=(20, 20))
        upload_icon = ctk.CTkImage(Image.open(f"{assets_dir}/upload_light.png"), size=(20, 20))
        ocr_icon = ctk.CTkImage(Image.open(f"{assets_dir}/ocr_light.png"), size=(20, 20))

        # Create buttons
        button_style = {"fg_color": self.colors["button"], "text_color": self.colors["text"], "hover_color": self.colors["accent"]}
        self.snap_button = ctk.CTkButton(buttons_frame, text="Capture", image=camera_icon, compound="left",
                                         command=self.capture_photo, **button_style)
        self.snap_button.pack(side="left", padx=(0, 5), expand=True, fill="x")

        self.upload_button = ctk.CTkButton(buttons_frame, text="Upload", image=upload_icon, compound="left",
                                           command=self.upload_image, **button_style)
        self.upload_button.pack(side="left", padx=5, expand=True, fill="x")

        self.ocr_button = ctk.CTkButton(buttons_frame, text="OCR", image=ocr_icon, compound="left",
                                        command=self.extract_text_from_image, **button_style)
        self.ocr_button.pack(side="left", padx=(5, 0), expand=True, fill="x")

        # Right frame for notes and image display
        right_frame = ctk.CTkFrame(tab, fg_color=self.colors["primary"])
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # Image display
        self.image_frame = ctk.CTkFrame(right_frame, fg_color=self.colors["secondary"])
        self.image_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.image_frame.pack_propagate(False)
        self.image_label = ctk.CTkLabel(self.image_frame, text='No image captured', image=None)
        self.image_label.pack(fill="both", expand=True, padx=5, pady=5)

        # Notes text area
        self.notes_frame = ctk.CTkFrame(right_frame, fg_color=self.colors["primary"])
        self.notes_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.notes_text = ctk.CTkTextbox(self.notes_frame, font=("Roboto", 12), wrap="word",
                                         fg_color=self.colors["secondary"], text_color=self.colors["text"])
        self.notes_text.pack(fill="both", expand=True)

        # Tags entry
        tags_frame = ctk.CTkFrame(right_frame, fg_color=self.colors["primary"])
        tags_frame.pack(fill="x", padx=10, pady=(0, 10))

        tags_label = ctk.CTkLabel(tags_frame, text="Tags:", font=("Roboto", 12, "bold"),
                                  text_color=self.colors["text"])
        tags_label.pack(side="left", padx=(0, 5))

        self.tags_entry = ctk.CTkEntry(tags_frame, placeholder_text="Enter tags (comma-separated)", font=("Roboto", 12),
                                       fg_color=self.colors["secondary"], text_color=self.colors["text"])
        self.tags_entry.pack(side="left", fill="x", expand=True)
   
    def setup_notes_tab(self, tab):
        # Search frame
        search_frame = ctk.CTkFrame(tab, fg_color=self.colors["primary"])
        search_frame.pack(fill="x", padx=10, pady=10)

        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="Search notes...", font=("Roboto", 12),
                                         fg_color=self.colors["secondary"], text_color=self.colors["text"])
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.search_entry.bind('<Return>', self.search_notes)

        search_button = ctk.CTkButton(search_frame, text="Search", command=self.search_notes,
                                      fg_color=self.colors["button"], text_color=self.colors["text"],
                                      hover_color=self.colors["accent"])
        search_button.pack(side="right")

        # Notes feed
        self.notes_canvas = ctk.CTkCanvas(tab, highlightthickness=0)
        self.notes_canvas.pack(side="left", fill="both", expand=True, padx=10, pady=(0, 10))

        scrollbar = ctk.CTkScrollbar(tab, orientation="vertical", command=self.notes_canvas.yview)
        scrollbar.pack(side="right", fill="y")

        self.notes_canvas.configure(yscrollcommand=scrollbar.set)
        self.notes_feed_frame = ctk.CTkFrame(self.notes_canvas, fg_color="transparent")
        self.notes_canvas.create_window((0, 0), window=self.notes_feed_frame, anchor="nw", width=1110)

        self.notes_feed_frame.bind("<Configure>", lambda e: self.notes_canvas.configure(scrollregion=self.notes_canvas.bbox("all")))

        # Load saved notes
        self.load_saved_notes()
    
    def setup_editor_tab(self):
        editor_frame = ctk.CTkFrame(self.notebook.tab("Editor"), fg_color=self.colors["primary"])
        editor_frame.pack(fill="both", expand=True, padx=10, pady=10)

        custom_font = tkfont.Font(family="Roboto", size=12)

        # Create the rich text editor
        self.editor = tk.Text(
            editor_frame,
            wrap="word",
            font=custom_font,
            undo=True,
            bg=self.colors["secondary"],
            fg=self.colors["text"],
            insertbackground=self.colors["accent"],
            selectbackground=self.colors["accent"],
            selectforeground=self.colors["text"],
            padx=10,
            pady=10,
            borderwidth=0,
            relief="flat"
        )
        self.editor.pack(fill="both", expand=True, padx=5, pady=5)

        # Create a toolbar for formatting options
        toolbar = ctk.CTkFrame(editor_frame, fg_color=self.colors["primary"])
        toolbar.pack(fill="x", pady=(0, 5))

        # Style for buttons
        button_style = {
            "fg_color": self.colors["button"],
            "text_color": self.colors["text"],
            "hover_color": self.colors["accent"],
            "corner_radius": 5,
            "width": 40,
            "height": 30
        }

        # Font family dropdown
        font_families = ["Arial", "Helvetica", "Times New Roman", "Courier", "Verdana"]
        self.font_family_var = tk.StringVar(value="Roboto")
        font_family_dropdown = ctk.CTkOptionMenu(toolbar, variable=self.font_family_var, values=font_families, 
                                                 command=self.change_font_family, 
                                                 fg_color=self.colors["button"],
                                                 button_color=self.colors["accent"],
                                                 button_hover_color=self.colors["text"],
                                                 dropdown_fg_color=self.colors["secondary"],
                                                 text_color=self.colors["text"],
                                                 width=120)
        font_family_dropdown.pack(side="left", padx=2)

        # Font size dropdown
        font_sizes = ["8", "10", "12", "14", "16", "18", "20", "24", "28", "36"]
        self.font_size_var = tk.StringVar(value="12")
        font_size_dropdown = ctk.CTkOptionMenu(toolbar, variable=self.font_size_var, values=font_sizes, 
                                               command=self.change_font_size, 
                                               fg_color=self.colors["button"],
                                               button_color=self.colors["accent"],
                                               button_hover_color=self.colors["text"],
                                               dropdown_fg_color=self.colors["secondary"],
                                               text_color=self.colors["text"],
                                               width=60)
        font_size_dropdown.pack(side="left", padx=2)

        # Bold button
        bold_button = ctk.CTkButton(toolbar, text="B", command=self.toggle_bold, **button_style)
        bold_button.pack(side="left", padx=2)

        # Italic button
        italic_button = ctk.CTkButton(toolbar, text="I", command=self.toggle_italic, **button_style)
        italic_button.pack(side="left", padx=2)

        # Underline button
        underline_button = ctk.CTkButton(toolbar, text="U", command=self.toggle_underline, **button_style)
        underline_button.pack(side="left", padx=2)

        # Text color button
        color_button = ctk.CTkButton(toolbar, text="A", command=self.change_text_color, **button_style)
        color_button.pack(side="left", padx=2)

        # Load voice icon
        assets_dir = "assets"
        voice_icon = ctk.CTkImage(Image.open(f"{assets_dir}/voice_light.png"), size=(20, 20))

        # Add the voice note button to the toolbar
        self.voice_button = ctk.CTkButton(toolbar, text="Voice Note", image=voice_icon, compound="left",
                                          command=self.toggle_voice_recording, **button_style)
        self.voice_button.pack(side="left", padx=2)
        
        # Add the save button to the toolbar
        save_button = ctk.CTkButton(toolbar, text="Save Note", command=self.save_note,
                                    fg_color=self.colors["accent"], text_color=self.colors["text"],
                                    hover_color=self.colors["button"], height=30, font=("Roboto", 12, "bold"))
        save_button.pack(side="right", padx=2)

    def change_font_family(self, family):
        self.editor.tag_add(f"family_{family}", "sel.first", "sel.last")
        self.editor.tag_configure(f"family_{family}", font=(family, int(self.font_size_var.get())))

    def change_font_size(self, size):
        self.editor.tag_add(f"size_{size}", "sel.first", "sel.last")
        self.editor.tag_configure(f"size_{size}", font=(self.font_family_var.get(), int(size)))

    def toggle_bold(self):
        current_tags = self.editor.tag_names("sel.first")
        if "bold" in current_tags:
            self.editor.tag_remove("bold", "sel.first", "sel.last")
        else:
            self.editor.tag_add("bold", "sel.first", "sel.last")
        self.editor.tag_configure("bold", font=tkfont.Font(weight="bold"))

    def toggle_italic(self):
        current_tags = self.editor.tag_names("sel.first")
        if "italic" in current_tags:
            self.editor.tag_remove("italic", "sel.first", "sel.last")
        else:
            self.editor.tag_add("italic", "sel.first", "sel.last")
        self.editor.tag_configure("italic", font=tkfont.Font(slant="italic"))

    def toggle_underline(self):
        current_tags = self.editor.tag_names("sel.first")
        if "underline" in current_tags:
            self.editor.tag_remove("underline", "sel.first", "sel.last")
        else:
            self.editor.tag_add("underline", "sel.first", "sel.last")
        self.editor.tag_configure("underline", underline=True)

    def change_text_color(self):
        color = colorchooser.askcolor(title="Choose text color")[1]
        if color:
            self.editor.tag_add(f"color_{color}", "sel.first", "sel.last")
            self.editor.tag_configure(f"color_{color}", foreground=color)
    
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
                # Get the current cursor position in the editor widget
                cursor_pos = self.editor.index(tk.INSERT)
                self.editor.insert(cursor_pos, f"\n{text}")
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
                image_label = ctk.CTkLabel(note_frame_inner, image=photo_image, text='')
                image_label.image = photo_image  # Keep a reference to prevent garbage collection
                image_label.pack(side="left", padx=5, pady=5)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                image_label = ctk.CTkLabel(note_frame_inner, text="No Image", width=100, height=100, fg_color="#E0E0E0")
                image_label.pack(side="left", padx=5, pady=5)

            text_frame = ctk.CTkFrame(note_frame_inner, fg_color=None)
            text_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

            # Display the note title
            note_title = ctk.CTkLabel(text_frame, text=note["title"], font=("Helvetica", 16, "bold"), anchor="w")
            note_title.pack(fill="x")

            # Display the note text
            note_text = ctk.CTkLabel(text_frame, text=note["note"], wraplength=520, anchor="w", justify="left")
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
        Tags: [tag1], [tag2], [tag3], ...[max 5 tags]
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
        Tags: [tag1], [tag2], [tag3], ...[max 5 tags]
        '''
        response = self.inference_chain.invoke(
            {"prompt": prompt, "image_base64": image_base64.decode()},
            config={"configurable": {"session_id": "unused"}}
        ).strip() 
        title, note, tags = self.parse_ai_response(response)
        self.display_note(image_base64, title, note, tags)
    
    def display_note(self, image_base64, title, note, tags):
        self.notes_text.delete("1.0", tk.END)
        self.editor.delete("1.0", tk.END)
        self.notes_text.insert(tk.END, f"{title}\n\n{note}\n\nTags: {', '.join(tags)}")
        self.editor.insert(tk.END, f"{title}\n\n", "title")
        self.editor.insert(tk.END, f"{note}\n\n")
        self.editor.insert(tk.END, f"Tags: {', '.join(tags)}", "tags")
        self.display_image(image_base64)
        self.tags_entry.delete(0, tk.END)
        self.tags_entry.insert(0, ', '.join(tags))
        
        # Apply styles
        self.editor.tag_configure("title", font=("Roboto", 16, "bold"), foreground=self.colors["accent"])
        self.editor.tag_configure("tags", font=("Roboto", 10, "italic"), foreground=self.colors["text"])

    def display_image(self, image_base64):
        image_array = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)  # Mirror the image horizontally for display
        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
        pil_image = Image.fromarray(image)
        photo_image = ctk.CTkImage(pil_image, size=(200, 200))
        self.image_label.configure(image=photo_image, text="")
        self.image_label.image = photo_image
    
    def update_camera_feed(self):
        frame = self.webcam_stream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480), interpolation=cv2.INTER_AREA)
        photo_image = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        if not hasattr(self, 'camera_image_id'):
            self.camera_image_id = self.canvas.create_image(375, 320, image=photo_image, anchor=tk.CENTER)
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
            imgbb_api_key = os.getenv('IMGBB_API_KEY')  # Make sure to set this environment variable
            image_url = self.upload_image_to_imgbb(image_path, imgbb_api_key)
            if image_url:
                blocks.append({
                    "object": "block",
                    "type": "image",
                    "image": {
                        "type": "external",
                        "external": {
                            "url": image_url
                        }
                    }
                })
            else:
                print("Failed to upload image, not including image block.")

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

    def upload_image_to_imgbb(self, image_path, api_key):
        with open(image_path, "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            payload = {
                "key": api_key,
                "image": base64.b64encode(file.read()).decode('utf-8'),
            }
            res = requests.post(url, payload)
            if res.status_code == 200:
                return res.json()['data']['url']
            else:
                print(f"Failed to upload image: {res.status_code} - {res.text}")
                return None
    
    def save_note(self):
        if self.last_note_image is None:
            logging.warning("No image to save.")
            return

        note = self.editor.get("1.0", tk.END).strip()
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

        for note in results:
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
                image_label = ctk.CTkLabel(note_frame_inner, image=photo_image, text='')
                image_label.image = photo_image  # Keep a reference to prevent garbage collection
                image_label.pack(side="left", padx=5, pady=5)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                image_label = ctk.CTkLabel(note_frame_inner, text="No Image", width=100, height=100, fg_color="#E0E0E0")
                image_label.pack(side="left", padx=5, pady=5)

            text_frame = ctk.CTkFrame(note_frame_inner, fg_color=None)
            text_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

            # Display the note title
            note_title = ctk.CTkLabel(text_frame, text=note["text"], font=("Helvetica", 16, "bold"), anchor="w")
            note_title.pack(fill="x")

            # Display the note text
            note_text = ctk.CTkLabel(text_frame, text=note["content"], wraplength=520, anchor="w", justify="left")
            note_text.pack(fill="x")

        # Update the canvas scrollregion
        self.notes_canvas.configure(scrollregion=self.notes_canvas.bbox("all"))
        
    def _create_inference_chain(self):
        SYSTEM_PROMPT = """
    You are an advanced note-taking assistant with image recognition capabilities. Your task is to create clear, structured, and informative notes based on the images provided. Adapt your note-taking style to the content of each image, whether it's a document, diagram, photograph, or any other visual input.

    Structure your notes as follows:

    1. Title: Create a concise, descriptive title (1 line)

    2. Main Subject Description (2-3 sentences):
    - Clearly identify and describe the primary subject or focus of the image
    - For code snippets or technical diagrams, provide a brief overview of their purpose or function

    3. Detailed Analysis (3-5 bullet points):
    - Break down key elements or components visible in the image
    - For text-heavy images, summarize main points or sections
    - For visual scenes, describe important details, spatial relationships, or notable features
    - For charts or graphs, explain the data representation and any significant trends

    4. Context and Significance (1-2 sentences):
    - Provide relevant background information or explain the importance of the image content
    - For educational or professional content, briefly mention its potential applications or implications

    5. Text Transcription (if applicable):
    - For images containing text, provide a concise summary or key excerpts
    - For code snippets, include important function names, class definitions, or critical lines

    6. Personal Insights or Action Items (1-2 bullet points):
    - Offer a brief interpretation, reflection, or potential follow-up actions based on the image content

    7. Tags: Generate 3-5 relevant tags that categorize the image content effectively [max 5 tags]

    Guidelines:
    - Maintain objectivity while providing insightful analysis
    - Use clear, concise language appropriate for the subject matter
    - Adapt the level of technical detail to match the complexity of the image content
    - Ensure your notes are informative enough to stand alone without the image
    - When uncertain about specific details, use phrases like "appears to be" or "likely represents" to indicate interpretation

    Remember, your goal is to create notes that are comprehensive yet easy to understand, catering to both quick reference and in-depth review.
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