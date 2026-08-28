import os
import uuid
from django.conf import settings
import pyttsx3
import threading

def generate_voice_file(text: str) -> str:
    """
    Generates a .wav file locally using pyttsx3.
    Returns the URL/path to the generated file.
    Runs in a dedicated thread to avoid COM initialization issues in Django.
    """
    media_dir = os.path.join(settings.MEDIA_ROOT, 'tts')
    os.makedirs(media_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.wav"
    filepath = os.path.join(media_dir, filename)
    url_path = f"{settings.MEDIA_URL}tts/{filename}"
    
    def run_synthesis():
        engine = pyttsx3.init()
        # Ensure it's a masculine voice by prioritizing David or male
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'David' in voice.name or 'male' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        
        # Make the voice calm and deliberate
        engine.setProperty('rate', 150)
        
        engine.save_to_file(text, filepath)
        engine.runAndWait()
        # important to stop/cleanup
        engine.stop()

    t = threading.Thread(target=run_synthesis)
    t.start()
    t.join() # Wait for generation to finish

    return url_path
