import pyaudio
import numpy as np


# Constants
CHANNELS = 1
SAMPLE_RATE = 16000
FORMAT = pyaudio.paInt16

class AudioInput:
    def __init__(self, channels=CHANNELS, sample_rate=SAMPLE_RATE, format=FORMAT, chunk_size=None):
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

#import numpy as np
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

import time
import torch
#import numpy as np

class VAD:
    def __init__(self, model_path, pause_threshold=2.0):
        self.model = torch.jit.load(model_path)
        self.model.eval()

        self.pause_threshold = pause_threshold
        self.last_speech_time = None
        self.current_state = "Silence"

    def vad(self, reduced_noise, sample_rate=16000):
        audio_float32 = NoiseReducer.int2float(reduced_noise)
        vad_confidence = self.model(torch.from_numpy(audio_float32), sample_rate).item()
        return vad_confidence

    def vad_states(self, vad_confidence):
        if vad_confidence > 0.6:  # Slightly higher threshold to start speaking
            self.last_speech_time = time.time()
            self.current_state = "Speaking"
        elif vad_confidence < 0.4:  # Slightly lower threshold to stop speaking
            if self.current_state == "Speaking":
                if self.last_speech_time and (time.time() - self.last_speech_time > self.pause_threshold):
                    self.current_state = "Silence"
                else:
                    self.current_state = "Pause"
            elif self.current_state == "Pause":
                if self.last_speech_time and (time.time() - self.last_speech_time > self.pause_threshold):
                    self.current_state = "Silence"
        else:
            # Handling the case where vad_confidence is between 0.4 and 0.6
            if self.current_state == "Speaking":
                self.current_state = "Speaking"
        
        return self.current_state



import wave

def test_vad(audio_input, noise_reducer, vad):
    print("Starting real-time audio recording and VAD processing. Press Ctrl+C to stop.")

    buffer = []          # Buffer to hold continuous audio
    file_index = 0       # Index to name the speech files
    segment_index = 1    # Index to track segments within a speech
    last_state = None    # Track the last state
    recorded_speech = False  # Flag to track if any speech has been recorded

    try:
        while True:
            audio_chunk = audio_input.record()  # Record audio in real-time
            reduced_noise = noise_reducer.reduce_noise(audio_chunk)  # Apply noise reduction
            vad_confidence = vad.vad(reduced_noise)  # Get VAD confidence
            vad_state = vad.vad_states(vad_confidence)  # Get VAD state

            if vad_state in ["Speaking", "Pause"]:
                buffer.append(audio_chunk)
                recorded_speech = True

            if last_state == "Pause" and vad_state == "Speaking":
                # Save the segment that includes the previous pause
                filename = f"audio_speech{file_index}_segment{segment_index}.wav"
                save_audio(buffer, filename, audio_input)
                buffer.clear()
                buffer.append(audio_chunk)  # Start new segment including the current chunk
                segment_index += 1

            elif vad_state == "Silence" and recorded_speech:
                if last_state == "Pause":
                    # If it ends in a pause, treat it as junk
                    filename = f"junk_audio_{file_index}.wav"
                else:
                    # Save as a regular speech segment
                    filename = f"audio_speech{file_index}_segment{segment_index}.wav"

                save_audio(buffer, filename, audio_input)
                buffer.clear()
                file_index += 1
                segment_index = 1
                recorded_speech = False  # Reset the flag

            last_state = vad_state  # Update the last state
            print(f"VAD Result: {vad_state} ({vad_confidence})")
    except KeyboardInterrupt:
        print("Stopping audio recording.")
        audio_input.close()

def save_audio(audio_chunks, filename, audio_input):
    """ Save a list of audio chunks to a WAV file """
    wf = wave.open(filename, 'wb')
    wf.setnchannels(audio_input.channels)
    wf.setsampwidth(audio_input.audio.get_sample_size(audio_input.format))
    wf.setframerate(audio_input.sample_rate)
    for chunk in audio_chunks:
        wf.writeframes(chunk)
    wf.close()
    print(f"Saved {filename}")

# Constants
MODEL_PATH = "P:/Assistant/Audio/silero-vad/files/silero_vad.jit"

# Initialize modules
audio_input = AudioInput()
noise_reducer = NoiseReducer()
vad = VAD(MODEL_PATH)


test_vad(audio_input, noise_reducer, vad)