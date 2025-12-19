import os
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

class ArtifactGenerator:
    def __init__(self, output_dir="artifacts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_visual_threat(self, text, uid):
        """Creates the typographic attack image."""
        filename = os.path.join(self.output_dir, f"{uid}_visual.png")
        
        # Skip generation if file already exists
        if os.path.exists(filename):
            return filename
        
        # Create canvas
        img = Image.new('RGB', (1024, 200), color='white')
        d = ImageDraw.Draw(img)
        
        # Use a generic font if system font fails
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except:
            font = ImageFont.load_default()
            
        d.text((20, 80), text, fill='black', font=font)
        img.save(filename)
        return filename

    def create_audio_mask(self, text, uid):
        """Creates the benign audio mask."""
        filename = os.path.join(self.output_dir, f"{uid}_audio.mp3")
        
        # Skip generation if file already exists
        if os.path.exists(filename):
            return filename
        
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        return filename