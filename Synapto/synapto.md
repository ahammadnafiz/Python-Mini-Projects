 Synapto - AI-Powered Note Taking App with Image Recognition

**Objective**:
Synapto is an intelligent note-taking application that leverages computer vision, OCR (Optical Character Recognition), and AI-based text generation to facilitate capturing and managing notes from images. The application is designed to be user-friendly, providing an integrated solution for taking photos, extracting text, and generating detailed notes.

**Key Features**:
1. **Real-Time Camera Feed**: 
   - Uses OpenCV to capture live video from the webcam.
   - Displays the video feed in the application using the Tkinter library.

2. **Image Capture and Upload**:
   - Allows users to capture photos directly from the webcam.
   - Users can also upload images from their local storage.

3. **AI-Powered Note Generation**:
   - Utilizes the Google Generative AI model to analyze captured or uploaded images.
   - Generates structured notes, including a title, description, additional details, personal reflections, and relevant tags.

4. **Optical Character Recognition (OCR)**:
   - Integrates Tesseract OCR to extract text from images.
   - Enhances note-taking by automatically including text found in images.

5. **Note Management**:
   - Provides an interface to display, edit, and save notes.
   - Supports searching through saved notes using keywords.
   - Allows tagging of notes for better organization and retrieval.

6. **User Interface**:
   - Built with the CustomTkinter (CTk) library for a modern and responsive UI.
   - Features buttons, text areas, and labels for user interactions.

**Technologies and Libraries Used**:
- **OpenCV**: For capturing and processing real-time video from the webcam.
- **Tkinter and CustomTkinter**: For building the graphical user interface.
- **PIL (Python Imaging Library)**: For handling image operations.
- **Base64**: For encoding and decoding image data.
- **Tesseract OCR**: For extracting text from images.
- **LangChain**: For integrating and managing interactions with the Google Generative AI model.
- **Threading**: To handle real-time video feed updates without freezing the UI.
- **Logging**: For debugging and tracking application events.

**Implementation Details**:
1. **WebcamStream Class**:
   - Manages the webcam feed, including starting and stopping the stream.
   - Captures frames and optionally encodes them in Base64 format.

2. **NoteTakingApp Class**:
   - Initializes the application GUI and handles user interactions.
   - Processes images through the AI model to generate notes.
   - Handles image upload, OCR extraction, and note saving/searching functionalities.

3. **AI Inference Chain**:
   - Utilizes LangChain to create a prompt template for generating notes based on images.
   - Integrates with the Google Generative AI model for sophisticated note generation.

**Setup and Usage**:
- Configure the `.env` file with the necessary API keys.
- Run the application using `python synapto.py`.

**Future Enhancements**:
- Adding support for different languages in OCR.
- Enhancing the AI model to provide more contextual and detailed notes.
- Integrating cloud storage options for saving and retrieving notes.
- Implementing voice recognition for note-taking and commands.

**Showcase**:
Synapto demonstrates the integration of multiple technologies to create a seamless and intelligent note-taking experience. The project showcases practical applications of AI, computer vision, and user interface design in a cohesive and user-friendly tool.
