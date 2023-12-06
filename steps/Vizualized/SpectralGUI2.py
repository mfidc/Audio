import pyaudio
import threading
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, lfilter, iirnotch, freqz
from matplotlib.animation import FuncAnimation
from tkinter import Tk, Button, Scale, HORIZONTAL


# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1024  # Size of audio chunk
WINDOW = 'hann'
NOVERLAP = 512  # Overlap between segments for the spectrogram
HISTORY_SECONDS = 3  # how many seconds of history to show

# Initialize deque with zeros
history_buffer = deque(np.zeros(CHUNK * int(SAMPLE_RATE / CHUNK * HISTORY_SECONDS)), 
                       maxlen=CHUNK * int(SAMPLE_RATE / CHUNK * HISTORY_SECONDS))

# Audio Stream Setup
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=SAMPLE_RATE, input=True,
                    frames_per_buffer=CHUNK)

# Figure Setup with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
fig.subplots_adjust(hspace=0.5)

# Spectrogram Subplot
im = ax1.imshow(np.zeros((int(NOVERLAP / 2 + 1), int(SAMPLE_RATE / CHUNK * 2))), 
                origin='lower', aspect='auto', extent=[0, 2, 0, SAMPLE_RATE / 2], 
                cmap='inferno', interpolation='none')
ax1.set_title('Real-time Spectrogram')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [Hz]')
ax1.set_ylim(0, 3400)  # Set y-limits to 0-2000 Hz
# Amplitude Subplot
time_vec = np.linspace(0, HISTORY_SECONDS, int(SAMPLE_RATE / CHUNK * HISTORY_SECONDS))
line, = ax2.plot(time_vec, np.zeros(int(SAMPLE_RATE / CHUNK * HISTORY_SECONDS)))
ax2.set_ylim(-32768, 32767)
ax2.set_xlim(0, HISTORY_SECONDS)
ax2.set_title('Amplitude Over Time')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Amplitude')

# Update Function for Animation
def update_fig(frame):
    global history_buffer

    # Read new data from the audio stream
    new_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    
    # Append new data to the history buffer
    history_buffer.extend(new_data)

    # Convert the buffer to a numpy array for processing
    long_data = np.array(history_buffer)

    # Generate the spectrogram data
    f, t, Sxx = spectrogram(long_data, fs=SAMPLE_RATE, window=WINDOW, nperseg=CHUNK, noverlap=NOVERLAP, mode='magnitude')
    
    # Update the spectrogram image
    im.set_data(10 * np.log10(Sxx + 1e-10))
    im.set_extent([0, HISTORY_SECONDS, 0, SAMPLE_RATE / 2])
    im.set_clim(vmin=np.percentile(Sxx, 1), vmax=np.percentile(Sxx, 99))

    # Update the amplitude plot without clearing the axes
    line.set_ydata(long_data[::-1])
    line.set_xdata(np.linspace(HISTORY_SECONDS, 0, len(long_data)))

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

    def threaded_update():
        while True:
            update_fig(None)


    def start_animation(self):
        def threaded_update():
            while True:
                update_fig(None)

        # Start the thread for real-time update
        self.update_thread = threading.Thread(target=threaded_update)
        self.update_thread.daemon = True  # Daemonize thread
        self.update_thread.start()

    def stop_animation(self):
        if self.update_thread.is_alive():
            self.update_thread.join()  # Wait for the thread to finish

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
