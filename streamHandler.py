import time
import threading
import sounddevice as sd

class StreamHandler:
    def __init__(self):
        """
        Base class for managing streams.
        """
        self.stream_active = False
        self.callback = None

    def start_stream(self, callback):
        """
        Start the stream with a callback function.
        :param callback: Function to handle streamed data.
        """
        if self.stream_active:
            print("Stream is already running.")
            return

        self.callback = callback
        self.stream_active = True
        threading.Thread(target=self._stream).start()

    def stop_stream(self):
        """
        Stop the stream.
        """
        if not self.stream_active:
            print("Stream is not running.")
            return

        self.stream_active = False
        print("Stream stopped.")

    def _stream(self):
        """
        Abstract method for streaming data. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TextStreamHandler(StreamHandler):
    def __init__(self, chunk_size=10, delay=0.5):
        """
        Handler for text streams.
        :param chunk_size: Number of characters per chunk.
        :param delay: Time interval between chunks (in seconds).
        """
        super().__init__()
        self.text_source = ""
        self.chunk_size = chunk_size
        self.delay = delay

    def set_source(self, text_source):
        """
        Set the source of text data for streaming.
        :param text_source: The text data to stream.
        """
        self.text_source = text_source

    def _stream(self):
        """
        Stream text data in chunks, invoking the callback for each chunk.
        """
        print("Text stream started.")
        start = 0
        while self.stream_active and start < len(self.text_source):
            chunk = self.text_source[start:start + self.chunk_size]
            if self.callback:
                self.callback(chunk)

            start += self.chunk_size
            time.sleep(self.delay)

        print("Text stream ended.")
        


class AudioStreamHandler(StreamHandler):
    def __init__(self):
        """
        Handler for audio streams.
        """
        super().__init__()
        self.devices = {}  # Stores devices with their stream objects
        self.streams = {}  # Dictionary to manage open streams

    def list_devices(self):
        """
        List available audio devices.
        """
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']}")

    def add_device(self, device_index, stream_type='input', sample_rate=44100, channels=2):
        """
        Add a new audio device.
        :param device_index: Index of the audio device.
        :param stream_type: Type of stream ('input' or 'output').
        :param sample_rate: Sample rate of the stream.
        :param channels: Number of channels.
        """
        if device_index in self.devices:
            print(f"Device {device_index} is already added.")
            return

        device = sd.query_devices(device_index)
        if stream_type == 'input':
            stream = sd.InputStream(
                device=device_index,
                samplerate=sample_rate,
                channels=channels
            )
        elif stream_type == 'output':
            stream = sd.OutputStream(
                device=device_index,
                samplerate=sample_rate,
                channels=channels
            )
        else:
            raise ValueError("Invalid stream type. Must be 'input' or 'output'.")

        self.devices[device_index] = {
            "name": device['name'],
            "stream_type": stream_type,
            "sample_rate": sample_rate,
            "channels": channels
        }
        self.streams[device_index] = stream

        print(f"Device '{device['name']}' added successfully.")

    def start_stream(self, device_index):
        """
        Start the audio stream for the given device.
        :param device_index: Index of the audio device.
        """
        if device_index not in self.streams:
            print(f"Device {device_index} is not added. Please add the device first.")
            return

        stream = self.streams[device_index]
        if not stream.active:
            stream.start()
            print(f"Stream for device {device_index} started.")
        else:
            print(f"Stream for device {device_index} is already running.")

    def stop_stream(self, device_index):
        """
        Stop the audio stream for the given device.
        :param device_index: Index of the audio device.
        """
        if device_index not in self.streams:
            print(f"Device {device_index} is not added.")
            return

        stream = self.streams[device_index]
        if stream.active:
            stream.stop()
            print(f"Stream for device {device_index} stopped.")
        else:
            print(f"Stream for device {device_index} is not active.")

    def stop_all_streams(self):
        """
        Stop all active streams.
        """
        for device_index, stream in self.streams.items():
            if stream.active:
                stream.stop()
                print(f"Stream for device {device_index} stopped.")

    def __del__(self):
        """
        Clean up by stopping all streams.
        """
        self.stop_all_streams()
