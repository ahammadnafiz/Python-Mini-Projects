import pyttsx3

def generate_voice(script):
    engine = pyttsx3.init()
    engine.save_to_file(script, 'output.wav')
    engine.runAndWait()
