import socket
import numpy as np
import pyaudio
import wave
import time

def receive_audio(path='./received_audio.wav',
                  HOST='192.168.1.2',  # Pico W's IP address
                  PORT=5000,
                  SAMPLE_RATE=16000,
                  CHANNELS=1,
                  FORMAT=pyaudio.paInt16,
                  CHUNK_SIZE=1600,
                  GRACE_PERIOD=5):  # Grace period in seconds
    """
    Receives audio data from the Pico W over TCP and saves it to a WAV file.
    Initially blocks to wait for data, then becomes non-blocking for termination.
    """
    # Each sample is 2 bytes (16 bits)
    BYTES_PER_SAMPLE = 2  # FIXED: Changed from 1 to 2 for 16-bit audio
    TOTAL_SAMPLES = SAMPLE_RATE * CHANNELS * 5  # 5 seconds of audio
    TOTAL_BYTES = TOTAL_SAMPLES * BYTES_PER_SAMPLE

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Create a stream to play audio
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    output=True,
                    frames_per_buffer=CHUNK_SIZE)  # Added frames_per_buffer

    frames = []  # List to store audio frames
    received_bytes = 0  # Counter for total bytes received
    first_byte_received = False
    last_data_time = time.time()  # Tracks time of last received data

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Connecting to {HOST}:{PORT}...")
        try:
            s.connect((HOST, PORT))
            print("Connected to Pico W.")
        except Exception as e:
            print(f"Failed to connect: {e}")
            return

        try:
            data_buffer = b''
            s.setblocking(True)  # Start with blocking mode
            print("Waiting for first byte...")

            while True:
                try:
                    # Receive data
                    data = s.recv(CHUNK_SIZE * BYTES_PER_SAMPLE)  # Adjusted receive size
                    if data:
                        if not first_byte_received:
                            first_byte_received = True
                            print("First byte received, switching to non-blocking mode.")
                            s.setblocking(False)  # Switch to non-blocking mode

                        received_bytes += len(data)
                        last_data_time = time.time()  # Reset the timeout timer
                        data_buffer += data

                        # Process data in CHUNK_SIZE increments
                        while len(data_buffer) >= CHUNK_SIZE * BYTES_PER_SAMPLE:  # Adjusted chunk check
                            chunk = data_buffer[:CHUNK_SIZE * BYTES_PER_SAMPLE]
                            data_buffer = data_buffer[CHUNK_SIZE * BYTES_PER_SAMPLE:]

                            # Convert bytes to NumPy array
                            audio_data = np.frombuffer(chunk, dtype=np.int16)

                            # Remove DC offset (optional)
                            dc_offset = np.mean(audio_data)
                            audio_data = audio_data - int(dc_offset)

                            # Apply gain to amplify the audio
                            gain_factor = 2.0
                            audio_data = audio_data * gain_factor

                            # Ensure we don't exceed the int16 range
                            audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)

                            # Convert back to bytes
                            processed_data = audio_data.tobytes()

                            # Write data to audio stream
                            stream.write(processed_data)

                            # Append data to frames list
                            frames.append(processed_data)

                        # Check if we have received enough data
                        if received_bytes >= TOTAL_BYTES:
                            print("Received enough audio data.")
                            break

                    else:
                        # Non-blocking termination if no data is received
                        if time.time() - last_data_time > GRACE_PERIOD:
                            print("Grace period exceeded, terminating.")
                            break

                except BlockingIOError:
                    # Non-blocking mode will raise this if no data is available
                    if time.time() - last_data_time > GRACE_PERIOD:
                        print("No more data available during grace period, terminating.")
                        break

        finally:
            print("Saving audio...")
            if frames:
                save_segment(frames, path, p, CHANNELS, FORMAT, SAMPLE_RATE)
            else:
                print("No frames captured.")
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Connection closed.")

def save_segment(frames, path, p, CHANNELS, FORMAT, SAMPLE_RATE):
    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio segment saved to {path}")

# Example usage
if __name__ == '__main__':
    receive_audio()