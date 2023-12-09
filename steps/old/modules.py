import pyaudio
import numpy as np

class AudioInput:
    def __init__(self, channels=1, sample_rate=16000, format=pyaudio.paInt16, chunk_size=None):
        self.channels = channels
        self.sample_rate = sample_rate
        self.format = format
        self.chunk_size = int(sample_rate / 4) if chunk_size is None else chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(input=True, format=self.format, rate=self.sample_rate, 
                                      channels=self.channels, frames_per_buffer=self.chunk_size)

    def record(self):
        return self.stream.read(self.chunk_size)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


import numpy as np
import noisereduce as nr

class NoiseReducer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    @staticmethod
    def int2float(sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / 32768
        sound = sound.squeeze()
        return sound

    def reduce_noise(self, audio_chunk):
        audio_int16 = np.frombuffer(audio_chunk, np.int16)
        reduced_noise = nr.reduce_noise(y=audio_int16, sr=self.sample_rate)
        return reduced_noise


import torch
import numpy as np

class VAD:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()

    def vad(self, reduced_noise, sample_rate=16000):
        audio_float32 = NoiseReducer.int2float(reduced_noise)
        vad_confidence = self.model(torch.from_numpy(audio_float32), sample_rate).item()
        return vad_confidence

    @staticmethod
    def vad_states(vad_confidence):
        if vad_confidence < 0.5:
            return "Silence"
        if vad_confidence > 0.5:
            return "Speaking"
        else:
            return False



# Constants
MODEL_PATH = "P:/Assistant/Audio/silero-vad/files/silero_vad.jit"

# Initialize modules
audio_input = AudioInput()
noise_reducer = NoiseReducer()
vad = VAD(MODEL_PATH)

try:
    while True:
        # Record audio
        audio_chunk = audio_input.record()

        # Noise Reduction
        reduced_noise = noise_reducer.reduce_noise(audio_chunk)

        # VAD
        vad_confidence = vad.vad(reduced_noise)
        state = VAD.vad_states(vad_confidence)

        # Process the state as needed
        print(state)
finally:
    audio_input.close()
