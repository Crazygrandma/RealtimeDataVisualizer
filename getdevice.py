import sounddevice as sd

# List all available devices
print("Available audio devices:")
print(sd.query_devices())

# Select a specific input device
# Replace 'Microphone (Your Device Name)' with the name or index of your input device
DEVICE_NAME = "CABLE Output (VB-Audio Virtual Cable)"  # Replace with the actual device name or index

# Ensure the device exists
device_list = sd.query_devices()
print(device_list)
device_indices = [i for i, d in enumerate(device_list) if DEVICE_NAME in d['name']]

if not device_indices:
    raise ValueError(f"Device '{DEVICE_NAME}' not found in the device list.")

DEVICE_INDEX = device_indices[0]
print(f"Using audio device: {device_list[DEVICE_INDEX]['name']} (Index: {DEVICE_INDEX})")

# Set the device in the InputStream
SAMPLERATE = 44100  # Adjust as needed

with sd.InputStream(device=4, samplerate=SAMPLERATE, channels=1) as stream:
    print("Recording with selected device... Press Ctrl+C to stop.")
    try:
        while True:
            pass  # Stream active
    except KeyboardInterrupt:
        print("\nStopped.")
