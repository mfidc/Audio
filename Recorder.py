import pyaudio
import numpy as np
import queue
import logging
import time

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
