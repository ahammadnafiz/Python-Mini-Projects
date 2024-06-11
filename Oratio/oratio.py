import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import os

from voice_assistant import process_voice_assistant
from vision_assistant import process_multimodel_assistant

# Create CustomTkinter window
root = ctk.CTk()
root.geometry("600x500")
root.title("Assistant")

# Get the background color of the Tkinter window
window_bg_color = root.cget("bg")

# Get the path to the Assets directory
Assets_dir = os.path.join(os.path.dirname(__file__), "Assets")

# Get the path to the microphone image file
voice_light_path = os.path.join(Assets_dir, "microphone.png")

# Open the image file
light_image = Image.open(voice_light_path)

# Create a transparent image with the same size as the original
transparent_image = Image.new("RGBA", light_image.size, (0, 0, 0, 0))

# Paste the original image onto the transparent image
transparent_image.paste(light_image, (0, 0), light_image)

# Resize the image to a smaller size
transparent_image = transparent_image.resize((120, 120), Image.Resampling.LANCZOS)

# Create a PhotoImage object from the PIL image
voice_button_image = ImageTk.PhotoImage(transparent_image)

# Create voice button
voice_button = tk.Button(root, image=voice_button_image, command=process_voice_assistant, bd=0, highlightthickness=0, bg=window_bg_color)
voice_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

# Create camera button
camera_button = ctk.CTkButton(root, text="Vision Assistant", command=process_multimodel_assistant)
camera_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

# Start the main event loop
root.mainloop()
