import time
import torch
import pyaudio
import numpy as np
import noisereduce as nr

#Constants
CHANNELS = 1
SAMPLE_RATE = 16000
FORMAT = pyaudio.paInt16
CHUNK = int(SAMPLE_RATE / 4) # 4 = 1/4 second

#Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(input=True, 
                    format=FORMAT,
                    rate=SAMPLE_RATE, 
                    channels=CHANNELS,
                    frames_per_buffer=CHUNK)

#Initialize VAD
model_path = "P:/Assistant/Audio/silero-vad/files/silero_vad.jit"
vad_model = torch.jit.load(model_path)
vad_model.eval()

def record():
    return stream.read(CHUNK)

def close():
    stream.stop_stream()
    stream.close()
    audio.terminate()

def int2float(sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / 32768
        sound = sound.squeeze()
        return sound

def reduce_noise(audio_chunk, sr=SAMPLE_RATE):
    # Convert the audio chunk to a NumPy array
    audio_int16 = np.frombuffer(audio_chunk, np.int16)

    # Apply noise reduction and return the NumPy array
    reduced_noise = nr.reduce_noise(y=audio_int16, sr=sr)
    return reduced_noise

def vad(reduced_noise, sr=SAMPLE_RATE):
    # Convert the reduced noise audio to float32 format
    audio_float32 = int2float(reduced_noise)

    # Perform VAD and return the result
    vad_confidence = vad_model(torch.from_numpy(audio_float32), sr).item()
    return vad_confidence

def vad_states(vad_confidence):
    if vad_confidence < 0.5:
        return "Silence"
    if vad_confidence > 0.5:
        return "Speaking"
    else:
        return False



import wave

def test_vad():
    print("Starting real-time audio recording and VAD processing. Press Ctrl+C to stop.")

    speech_buffer = []  # Buffer to hold audio during speech
    is_speech = False   # Flag to track whether current segment is speech
    file_index = 0      # Index to name the output files

    try:
        while True:
            audio_chunk = record()  # Record audio in real-time
            reduced_noise = reduce_noise(audio_chunk)  # Apply noise reduction
            vad_confidence = vad(reduced_noise)  # Get VAD confidence
            vad_state = vad_states(vad_confidence)  # Get VAD state

            if vad_state == "Speaking":
                is_speech = True
                speech_buffer.append(audio_chunk)
            elif is_speech and vad_state == "Silence":
                # Save the speech segment to a file
                filename = f"speech_segment_{file_index}.wav"
                save_audio(speech_buffer, filename)
                file_index += 1

                # Reset for the next speech segment
                is_speech = False
                speech_buffer.clear()

            print(f"VAD Result: {vad_state} ({vad_confidence})")
    except KeyboardInterrupt:
        print("Stopping audio recording.")
        close()

def save_audio(audio_chunks, filename):
    """ Save a list of audio chunks to a WAV file """
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    for chunk in audio_chunks:
        wf.writeframes(chunk)
    wf.close()
    print(f"Saved {filename}")

# Run the test
test_vad()