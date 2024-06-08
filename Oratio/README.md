```markdown
# Oratio: Advanced Voice and Vision Assistant

Oratio is an advanced application that serves as a versatile Voice and Vision Assistant, integrating sophisticated AI models for enriched user interaction. Built upon CustomTkinter for an elegant UI, Oratio offers seamless voice command recognition and vision functionalities, making it an indispensable tool for diverse applications.

## Features

- **Voice Assistant**: Responds intelligently to user voice commands, utilizing advanced speech recognition and a dynamic text-to-speech engine.
- **Vision Assistant**: Harnesses the power of AI to process webcam video input, providing insightful analysis and real-time assistance.

## Installation

### Prerequisites

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/)

### Clone the repository

```sh
git clone https://github.com/yourusername/oratio.git
cd Oratio
```

### Install dependencies

```sh
pip install -r requirements.txt
```

### Set up environment variables

Create a `.env` file in the project root and add your API keys:

```env
API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage

### Running the application

```sh
python oratio.py
```

### Main window

Oratio's main window presents intuitive buttons to access its Voice and Vision Assistant functionalities.

- **Voice Assistant**: Click to activate the voice assistant.
- **Vision Assistant**: Click to start the vision assistant.

## Project Structure

```
oratio/
│
├── vision_assistant.py              # Code for the Assistant class
├── requirements.txt          # Required Python packages
├── oradio.py                   # Entry point of the application
├── README.md                 # This README file
└── voice_assistant.py        # Code for the voice assistant
```

## Dependencies

- `customtkinter`: For building a sleek UI.
- `dotenv`: For loading environment variables.
- `opencv-python`: For webcam access and image processing.
- `Pillow`: For image manipulation.
- `pyttsx3`: For text-to-speech conversion.
- `SpeechRecognition`: For speech recognition capabilities.
- `langchain`: For integrating AI models.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Contact

For questions or suggestions, feel free to open an issue or contact me at ahammadnafiz@outlook.com.

```
