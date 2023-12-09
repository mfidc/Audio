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
        if vad_confidence > 0.5:  # Slightly higher threshold to start speaking
            self.last_speech_time = time.time()
            self.current_state = "Speaking"
        elif vad_confidence < 0.5:  # Slightly lower threshold to stop speaking
            if self.current_state == "Speaking":
                if self.last_speech_time and (time.time() - self.last_speech_time > self.pause_threshold):
                    self.current_state = "Silence"
                else:
                    self.current_state = "Pause"
            elif self.current_state == "Pause":
                if self.last_speech_time and (time.time() - self.last_speech_time > self.pause_threshold):
                    self.current_state = "Silence"
        else:
           
            if self.current_state == "Speaking":
                self.current_state = "Speaking"
        
        return self.current_state

import io
from tempfile import NamedTemporaryFile 
import speech_recognition as sr
from faster_whisper import WhisperModel 

class Transcriber:
    def __init__(self, model_type="large", device_type="cuda", compute_type="float16", cpu_threads=0):
        if model_type == "large":
            model_type = "large-v3"
        self.audio_model = WhisperModel(model_type, device=device_type, compute_type=compute_type, cpu_threads=cpu_threads)
        self.temp_file = NamedTemporaryFile().name

    def transcribe_audio(self, audio_sample, sample_rate, sample_width):
        audio_data = sr.AudioData(audio_sample, sample_rate, sample_width)
        wav_data = io.BytesIO(audio_data.get_wav_data())

        with open(self.temp_file, 'w+b') as f:
            f.write(wav_data.read())

        segments, _ = self.audio_model.transcribe(self.temp_file)
        text = ''.join(segment.text for segment in segments)
        return text

import wave

def test_vad(audio_input, noise_reducer, vad):
    print("Starting real-time audio recording and VAD processing. Press Ctrl+C to stop.")

    buffer = []  # Buffer for continuous audio
    pause_buffer = []  # Separate buffer for pause audio
    file_index = 0  # Index for naming speech files
    segment_index = 1  # Index for tracking segments within a speech
    last_state = None  # Track the last state
    transcriber = Transcriber()

    try:
        while True:
            audio_chunk = audio_input.record()
            reduced_noise = noise_reducer.reduce_noise(audio_chunk)
            vad_confidence = vad.vad(reduced_noise)
            vad_state = vad.vad_states(vad_confidence)

            # Handling buffer based on VAD state
            if vad_state in ["Speaking", "Pause"]:
                buffer.append(audio_chunk)
                if vad_state == "Pause":
                    pause_buffer.append(audio_chunk)

            # State transition from Speaking to Pause or Silence
            if last_state == "Speaking" and vad_state in ["Pause", "Silence"]:
                save_audio(buffer, f"audio_speech{file_index}_segment{segment_index}.wav", audio_input, transcriber)
                buffer.clear()
                segment_index += 1
                if vad_state == "Pause":
                    pause_buffer.clear()
                    pause_buffer.append(audio_chunk)

            # State transition from Pause to Silence
            elif last_state == "Pause" and vad_state == "Silence":
                #save_audio(pause_buffer, f"junk_audio_{file_index}.wav", audio_input, transcriber)
                pause_buffer.clear()
                file_index += 1
                segment_index = 1

            last_state = vad_state
            #print(f"VAD Result: {vad_state} ({vad_confidence})")

    except KeyboardInterrupt:
        print("Stopping audio recording.")
        audio_input.close()


def save_audio(audio_chunks, filename, audio_input, transcriber):
    """ Save a list of audio chunks to a WAV file and transcribe them """
    # Combine audio chunks
    combined_audio = b''.join(audio_chunks)

    # Transcription
    transcription = transcriber.transcribe_audio(combined_audio, audio_input.sample_rate, audio_input.audio.get_sample_size(audio_input.format))
    print(transcription)

    # Saving audio
    wf = wave.open(filename, 'wb')
    wf.setnchannels(audio_input.channels)
    wf.setsampwidth(audio_input.audio.get_sample_size(audio_input.format))
    wf.setframerate(audio_input.sample_rate)
    wf.writeframes(combined_audio)
    wf.close()
    #print(f"Saved {filename}")

# Constants
MODEL_PATH = "P:/Assistant/Audio/silero-vad/files/silero_vad.jit"

# Initialize modules
audio_input = AudioInput()
noise_reducer = NoiseReducer()
vad = VAD(MODEL_PATH)


test_vad(audio_input, noise_reducer, vad)