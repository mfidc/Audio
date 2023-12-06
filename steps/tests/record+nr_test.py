import time
import queue
import logging
import numpy as np
import pyaudio
import soundfile as sf
import noisereduce as nr

# Load noise profile
noise_data, rate = sf.read('noise.wav')
noise_profile = noise_data[:rate]
#noise_profile = None




def reduce_noise(audio_chunk, noise_profile, rate):
    # Convert to float32 for noise reduction
    audio_chunk_float = audio_chunk.astype(np.float32)
    reduced_noise_chunk = nr.reduce_noise(y=audio_chunk_float, sr=rate, )
    # Convert back to int16
    return reduced_noise_chunk.astype(np.int16)

class AudioCapture:
    def __init__(self, rate=16000, chunk_size=1024, process_callback=None):
        self.rate = rate
        self.chunk_size = chunk_size
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self.process_callback = process_callback
        self.data_queue = queue.Queue()
        self.running = False

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def audio_callback(self, in_data, frame_count, time_info, status):
        start_time = time.time()
        
        try:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            if self.process_callback:
                processed_data = self.process_callback(audio_data)
            else:
                processed_data = audio_data
            self.data_queue.put(processed_data)
        except Exception as e:
            logging.error(f"Error in audio callback: {e}")
            # Attempt to recover or notify for manual intervention
            return (None, pyaudio.paAbort)

        processing_time = time.time() - start_time
        logging.info(f"Audio processing time: {processing_time:.2f} seconds")

        if self.data_queue.qsize() > 10:  # Example threshold
            logging.warning("Data queue size is large, indicating a potential bottleneck.")

        return (in_data, pyaudio.paContinue)

    def start_stream(self):
        self.running = True
        self.stream = self.audio_interface.open(format=pyaudio.paInt16,
                                                channels=1,
                                                rate=self.rate,
                                                input=True,
                                                frames_per_buffer=self.chunk_size,
                                                stream_callback=self.audio_callback)
        self.stream.start_stream()

    def read_data(self):
        return self.data_queue.get()

    def stop_stream(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio_interface.terminate()

    def is_running(self):
        return self.running


import wave

def normalize_audio(reduced_noise_chunk):
    # Normalize the audio chunk within the int16 range
    max_val = np.iinfo(np.int16).max
    #audio_chunk = (reduced_noise_chunk / np.max(np.abs(reduced_noise_chunk))) * max_val
    return reduced_noise_chunk.astype(np.int16)

def test_audio_capture(capture_duration=10, rate=16000, chunk_size=1024):
    audio_capture = AudioCapture(rate=rate, chunk_size=chunk_size, process_callback=normalize_audio)
    audio_capture.start_stream()

    frames = []
    start_time = time.time()

    while time.time() - start_time < capture_duration:
        data = audio_capture.read_data()
        frames.append(data.tobytes())

    audio_capture.stop_stream()

    # Save the recorded data as a WAV file
    with wave.open('test_audio.wav', 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio_capture.audio_interface.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print("Test recording saved to 'test_audio.wav'")

if __name__ == "__main__":
    test_audio_capture()