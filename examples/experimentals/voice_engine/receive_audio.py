import socket
import threading
import pyaudio
import wave
import time
import os

# Replace with your Pico W's IP address and port
HOST = '192.168.1.24'  # Pico W's IP address
PORT = 5000

# Audio settings
SAMPLE_RATE = 16000  # Match with Pico's sample rate
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024  # Number of bytes per chunk

# Calculate the number of chunks needed for a 5-second segment
SAMPLES_PER_CHUNK = CHUNK_SIZE // 2
DURATION_PER_CHUNK = SAMPLES_PER_CHUNK / SAMPLE_RATE  # Duration of each chunk in seconds
CHUNKS_PER_SEGMENT = int(5 / DURATION_PER_CHUNK)  # Number of chunks for 5 seconds

# Timeout for no data
NO_DATA_TIMEOUT = 1  # 1 second

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
    last_received_time = time.time()  # Track the last received time

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Connecting to {HOST}:{PORT}...")
        s.connect((HOST, PORT))
        print("Connected to Pico W. Receiving audio data...")

        try:
            while True:
                data = s.recv(CHUNK_SIZE)
                current_time = time.time()

                if data:
                    print(f"Received {len(data)} bytes of data")
                    # Reset the last received time
                    last_received_time = current_time
                    # Write data to audio stream
                    stream.write(data)
                    # Append data to frames list for continuous playback
                    frames.append(data)
                    # Append data to segment_frames
                    segment_frames.append(data)

                    # Check if we've collected enough chunks for a 5-second segment
                    if len(segment_frames) >= CHUNKS_PER_SEGMENT:
                        # Save the segment to a WAV file
                        save_segment(segment_frames, segment_number)
                        segment_frames = []  # Reset segment frames
                        segment_number += 1  # Increment segment counter
                else:
                    # No data received
                    if current_time - last_received_time > NO_DATA_TIMEOUT:
                        print("No data received for more than 1 second.")
                        # Save the entire audio data
                        save_audio(frames)
                        # Delete all saved segments
                        delete_segments(segment_number)
                        break
        except KeyboardInterrupt:
            pass
        finally:
            print("Closing connection...")
            s.close()
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Connection closed.")

def save_segment(frames, segment_number):
    output_filename = f'received_audio_segment_{segment_number}.wav'
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Segment {segment_number} saved to {output_filename}")

def save_audio(frames):
    output_filename = 'received_audio.wav'
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio data saved to {output_filename}")

def delete_segments(segment_count):
    for i in range(1, segment_count):
        segment_filename = f'received_audio_segment_{i}.wav'
        if os.path.exists(segment_filename):
            os.remove(segment_filename)
            print(f"Deleted segment {segment_filename}")

if __name__ == '__main__':
    receive_audio()