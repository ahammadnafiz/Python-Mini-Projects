import tkinter as tk
import customtkinter as ctk

from voice_assistant import process_voice_assistant
from vision_assistant import process_multimodel_assistant


# Create CustomTkinter window
root = ctk.CTk()
root.geometry("600x400")
root.title("Assistant")

# Create voice button
voice_button = ctk.CTkButton(root, text="Voice Assistant", command=process_voice_assistant)
voice_button.place(relx=0.3, rely=0.5, anchor=tk.CENTER)

# Create camera button
camera_button = ctk.CTkButton(root, text="Camera Assistant", command=process_multimodel_assistant)
camera_button.place(relx=0.7, rely=0.5, anchor=tk.CENTER)

# Start the main event loop
root.mainloop()