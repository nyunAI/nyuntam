import socket
import numpy as np
import pyaudio
import wave
import time
import os

def receive_audio(path,
                  HOST='192.168.1.2',  # Pico W's IP address
                  PORT=5000,
                  SAMPLE_RATE=16000,
                  CHANNELS=1,
                  FORMAT=pyaudio.paInt16,
                  CHUNK_SIZE=1600):
    """
    Receives audio data from the Pico W over TCP and saves it to a WAV file.

    Parameters:
    - path (str): The file path where the audio will be saved.
    - HOST (str): The IP address of the Pico W. Default is '192.168.1.2'.
    - PORT (int): The port number to connect to on the Pico W. Default is 5000.
    - SAMPLE_RATE (int): The audio sample rate. Default is 16000 Hz.
    - CHANNELS (int): The number of audio channels. Default is 1.
    - FORMAT: The audio format. Default is pyaudio.paInt16.
    - CHUNK_SIZE (int): The size of each audio chunk in bytes. Default is 1600.

    Returns:
    - bytes: The 5 seconds of audio data received.
    """
    # Each sample is 2 bytes (16 bits)
    SAMPLES_PER_CHUNK = CHUNK_SIZE // 2
    DURATION_PER_CHUNK = SAMPLES_PER_CHUNK / SAMPLE_RATE  # Duration of each chunk in seconds
    CHUNKS_PER_SEGMENT = int(5 / DURATION_PER_CHUNK)  # Number of chunks for 5 seconds

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Create a stream to play audio
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    output=True,
                    frames_per_buffer=SAMPLES_PER_CHUNK)
    
    frames = []  # List to store audio frames for playback and saving

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Connecting to {HOST}:{PORT}...")
        try:
            s.connect((HOST, PORT))
            print("Connected to Pico W. Receiving audio data...")
        except Exception as e:
            print(f"Failed to connect: {e}")
            return None

        try:
            chunks_received = 0
            while chunks_received < CHUNKS_PER_SEGMENT:
                data = s.recv(CHUNK_SIZE)
                if not data:
                    print("No data received. Connection may have closed.")
                    break
                print(f"Received {len(data)} bytes of data")

                # Convert bytes to NumPy array
                audio_data = np.frombuffer(data, dtype=np.int16)

                # Remove DC offset (if necessary)
                dc_offset = np.mean(audio_data)
                audio_data = audio_data - int(dc_offset)

                # Apply gain to amplify the audio
                gain_factor = 2.0  # Adjust this value as needed
                audio_data = audio_data * gain_factor

                # Ensure we don't exceed the int16 range
                audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)

                # Convert back to bytes
                processed_data = audio_data.tobytes()

                # Write data to audio stream
                stream.write(processed_data)

                # Append data to frames list
                frames.append(processed_data)

                chunks_received += 1

            # Save the segment to a WAV file
            save_segment(frames, path, p, CHANNELS, FORMAT, SAMPLE_RATE)

            # Return the 5 seconds of audio bytes received
            audio_bytes = b''.join(frames)
            return audio_bytes

        except KeyboardInterrupt:
            pass
        finally:
            print("Closing connection...")
            s.close()
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Connection closed.")

def save_segment(frames, path, p, CHANNELS, FORMAT, SAMPLE_RATE):
    output_filename = path
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio segment saved to {output_filename}")

# Example usage:
if __name__ == '__main__':
    # Specify the path where the WAV file should be saved
    path = 'received_audio_segment.wav'
    while(1):
        audio_bytes = receive_audio(path)
        time.sleep(2)
    # Now audio_bytes contains the 5 seconds of audio data received
