import pyaudio
import numpy as np
import torch


#Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 6.67) # 1 second / 6.67 = 150ms
VAD_PATH = "P:/vad/silero-vad/files/silero_vad.jit"


class VADProcessor:
    def __init__(self, model_path):
        self.vad_model = torch.jit.load(model_path)
        self.vad_model.eval()
    
    @staticmethod
    def int2float(sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / 32768
        sound = sound.squeeze()
        return sound
    
    def process(self, audio_chunk):
        audio_int16 = np.frombuffer(audio_chunk, np.int16)
        audio_float32 = self.int2float(audio_int16)
        with torch.no_grad():
            return self.vad_model(torch.tensor(audio_float32).unsqueeze(0), SAMPLE_RATE).item()
        


#initialize pyaudio
audio = pyaudio.PyAudio()
vad_processor = VADProcessor(VAD_PATH)

#open stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

try:
    print("Recording... Speak into the microphone.")
    while True:
        data = stream.read(CHUNK)
        vad_confidence = vad_processor.process(data)
        print(f"VAD Confidence: {vad_confidence}")

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # Stop and close the stream and audio
    stream.stop_stream()
    stream.close()
    audio.terminate()