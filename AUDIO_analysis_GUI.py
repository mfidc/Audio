import pyaudio
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import librosa

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 6.67)  # 150ms
DISPLAY_SECONDS = 30  # Number of seconds of data to display
MAX_SAMPLES = int(DISPLAY_SECONDS * (1000 / 150))  # Number of chunks to keep
VAD_PATH = "P:/vad/silero-vad/files/silero_vad.jit"

# VAD Processor Class
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
        

def spectral_analysis(audio_data):
    # Convert to the frequency domain
    S = np.abs(librosa.stft(audio_data))
    spectral_centroids = librosa.feature.spectral_centroid(S=S)[0]
    return np.mean(spectral_centroids)


# GUI Class
class VADGui:
    def __init__(self, master):
        self.master = master
        master.title("Audio Analysis")

        # Plotting setup for VAD Confidence
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.vad_processor = VADProcessor(VAD_PATH)
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

        # Data storage for plots
        self.confidences = []
        self.spectral_data = []

        # Start the plotting process
        self.update_plot()

    def update_plot(self):
        try:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            vad_confidence = self.vad_processor.process(data)
            self.confidences.append(vad_confidence)
            self.confidences = self.confidences[-MAX_SAMPLES:]  # Keep only recent samples

            # Spectral Analysis
            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = VADProcessor.int2float(audio_int16)
            spectral_value = spectral_analysis(audio_float32)
            self.spectral_data.append(spectral_value)
            self.spectral_data = self.spectral_data[-MAX_SAMPLES:]  # Keep only recent samples

            # Update VAD Confidence plot
            self.ax1.clear()
            self.ax1.plot(self.confidences, label='VAD Confidence', color='blue')
            self.ax1.legend(loc='upper right')
            self.ax1.set_ylim(0, 1)
            self.ax1.set_ylabel('Confidence')

            # Update Spectral Centroid plot
            self.ax2.clear()
            self.ax2.plot(self.spectral_data, label='Spectral Centroid', color='orange')
            self.ax2.legend(loc='upper right')
            self.ax2.set_ylabel('Hz')
            self.ax2.set_xlabel('Samples')

            # Redraw the canvas
            self.canvas.draw()
        except Exception as e:
            print(f"Error: {e}")

        self.master.after(1, self.update_plot)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.master.destroy()

# Main function
def main():
    root = tk.Tk()
    gui = VADGui(root)
    root.protocol("WM_DELETE_WINDOW", gui.close)
    root.mainloop()

if __name__ == "__main__":
    main()
