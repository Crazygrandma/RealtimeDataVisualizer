from streamHandler import AudioStreamHandler, TextStreamHandler

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")
    print(f"Audio")

def text_callback(chunk):
    print(chunk)

if __name__ == "__main__":
    try:
        # Initialize handlers
        audio_handler = AudioStreamHandler()
        text_handler = TextStreamHandler(chunk_size=3, delay=0.2)

        # # Text Streaming
        text_data = "This is a simulated stream of text data split into chunks."
        text_handler.set_source(text_data)
        
        # Audio Streaming
        print("\nAvailable audio devices:")
        audio_handler.list_devices()
        device_index = int(input("\nEnter the device index for audio: "))
        audio_handler.add_device(device_index=device_index, stream_type='input')
        
        
        # Start the streams
        text_handler.start_stream(callback=text_callback)
        audio_handler.start_stream(device_index)

        print("\nStreaming text and audio. Press Ctrl+C to stop.")
        while True:
            pass

    except KeyboardInterrupt:
        print("\nStopping all streams.")
        text_handler.stop_stream()
        audio_handler.stop_all_streams()
