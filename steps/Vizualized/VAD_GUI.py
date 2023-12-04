import pyaudio
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 6.67)  # 150ms
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

# GUI Class
class VADGui:
    def __init__(self, master):
        self.master = master
        master.title("VAD Confidence Plot")

        # Plotting setup
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.vad_processor = VADProcessor(VAD_PATH)
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

        self.confidences = []
        self.update_plot()

    def update_plot(self):
        try:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            vad_confidence = self.vad_processor.process(data)
            self.confidences.append(vad_confidence)

            self.ax.clear()
            self.ax.plot(self.confidences, label='VAD Confidence')
            self.ax.legend()
            self.ax.set_ylim(0, 1)
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
