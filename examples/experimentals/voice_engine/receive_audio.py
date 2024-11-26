import socket
import threading
import pyaudio
import wave
import time

# Replace with your Pico W's IP address and port
HOST = '192.168.1.24'  # Pico W's IP address
PORT = 5000

# Audio settings
SAMPLE_RATE = 16000  # Match with Pico's sample rate
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024  # Number of bytes per chunk

# Calculate the number of chunks needed for a 3-second segment
# Each sample is 2 bytes (16 bits), so samples per chunk = CHUNK_SIZE / 2
SAMPLES_PER_CHUNK = CHUNK_SIZE // 2
DURATION_PER_CHUNK = SAMPLES_PER_CHUNK / SAMPLE_RATE  # Duration of each chunk in seconds
CHUNKS_PER_SEGMENT = int(3 / DURATION_PER_CHUNK)  # Number of chunks for 3 seconds

# Initialize PyAudio
p = pyaudio.PyAudio()

# Create a stream to play audio
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE)

def receive_audio():
    frames = []  # List to store audio frames for continuous playback
    segment_frames = []  # List to store frames for the current segment
    segment_number = 1  # Counter for segment files

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Connecting to {HOST}:{PORT}...")
        s.connect((HOST, PORT))
        print("Connected to Pico W. Receiving audio data...")

        try:
            while True:
                data = s.recv(CHUNK_SIZE)
                if not data:
                    print("No data received. Connection may have closed.")
                    break
                print(f"Received {len(data)} bytes of data")
                # Write data to audio stream
                stream.write(data)
                # Append data to frames list for continuous playback
                frames.append(data)
                # Append data to segment_frames
                segment_frames.append(data)

                # Check if we've collected enough chunks for a 3-second segment
                if len(segment_frames) >= CHUNKS_PER_SEGMENT:
                    # Save the segment to a WAV file
                    save_segment(segment_frames, segment_number)
                    segment_frames = []  # Reset segment frames
                    segment_number += 1  # Increment segment counter

        except KeyboardInterrupt:
            pass
        finally:
            print("Closing connection...")
            s.close()
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Connection closed.")

            # Optionally, save the entire audio data to a WAV file
            save_audio(frames)

def save_segment(frames, segment_number):
    output_filename = f'received_audio_segment_{segment_number}.wav'
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Segment {segment_number} saved to {output_filename}")

# Optionally, define a function to save the entire audio data
def save_audio(frames):
    output_filename = '/data/data/com.termux/files/home/user-input-audio/received_audio.wav'
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio data saved to {output_filename}")

if __name__ == '__main__':
    receive_audio()
