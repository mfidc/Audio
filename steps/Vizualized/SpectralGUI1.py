import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram
from tkinter import Tk, Button, Scale, HORIZONTAL

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1024  # Size of audio chunk
WINDOW = 'hann'
NOVERLAP = 512  # Overlap between segments for the spectrogram

# Audio Stream Setup
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=SAMPLE_RATE, input=True,
                    frames_per_buffer=CHUNK)

# Figure Setup with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.tight_layout(pad=3.0)

# Spectrogram Subplot
im = ax1.imshow(np.zeros((int(NOVERLAP / 2 + 1), int(CHUNK / NOVERLAP))), 
                origin='lower', aspect='auto', extent=[0, CHUNK / SAMPLE_RATE, 0, SAMPLE_RATE / 2], 
                cmap='inferno')
ax1.set_title('Real-time Spectrogram')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [Hz]')

# Amplitude Subplot
time_vec = np.linspace(0, CHUNK / SAMPLE_RATE, CHUNK)
line, = ax2.plot(time_vec, np.zeros(CHUNK))
ax2.set_ylim(-32768, 32767)
ax2.set_title('Amplitude Over Time')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Amplitude')

# Update Function for Animation
def update_fig(frame):
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    f, t, Sxx = spectrogram(data, fs=SAMPLE_RATE, window=WINDOW, nperseg=CHUNK, noverlap=NOVERLAP, mode='magnitude')
    im.set_data(10 * np.log10(Sxx + 1e-10))  # Avoid log of zero
    im.set_clim(vmin=np.percentile(Sxx, 1), vmax=np.percentile(Sxx, 99))
    
    line.set_ydata(data)
    return im, line

# GUI Class with enhanced features
class SpectrogramGUI:
    def __init__(self, master):
        self.master = master
        master.title("Real-Time Audio Analysis")

        self.start_button = Button(master, text="Start", command=self.start_animation)
        self.start_button.pack()

        self.stop_button = Button(master, text="Stop", command=self.stop_animation)
        self.stop_button.pack()

        # Dynamic Range Control
        self.dynamic_range = Scale(master, from_=0, to=100, orient=HORIZONTAL, label='Dynamic Range')
        self.dynamic_range.set(50)  # Default value
        self.dynamic_range.pack()

        # Animation with the update function
        self.animation = FuncAnimation(fig, update_fig, interval=50, save_count=50)

    def start_animation(self):
        self.animation.event_source.start()

    def stop_animation(self):
        self.animation.event_source.stop()

    def close(self):
        if self.animation is not None:
            self.animation.event_source.stop()
        stream.stop_stream()
        stream.close()
        audio.terminate()
        plt.close(fig)
        self.master.destroy()

# Main Function
def main():
    root = Tk()
    gui = SpectrogramGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.close)
    plt.show()

if __name__ == "__main__":
    main()
