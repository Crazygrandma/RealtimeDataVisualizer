import numpy as np
import sounddevice as sd
import scipy.fftpack
import cv2
import time

from audioDecoders import RandomDecoder

# Parameters
DURATION = 0.1 # Duration of each audio capture in seconds
LATENT_DIM = 32  # Dimension of the latent space
NOISE_THRESHOLD = 0.001  # Adjust based on your environment (calibrate for silence)
IMAGE_SIZE = 256  # Output image size (64x64)



# Initialize the random decoder
random_decoder = RandomDecoder(LATENT_DIM, IMAGE_SIZE)

# Random projection matrix (audio features to latent space)
np.random.seed(12)  # Ensure reproducibility
random_projection = np.random.randn(1024, LATENT_DIM)

# Initialize OpenCV for real-time display
# cv2.namedWindow("Generated Image", cv2.WINDOW_NORMAL)

# Global variable to store the latest generated image
latest_image = None

def display_image():
    """
    Separate thread to display the image without freezing the audio callback.
    """
    global latest_image
    while True:
        if latest_image is not None:
            # print(latest_image)
            cv2.imshow("Generated Image", latest_image)
            cv2.waitKey(1)  # Refresh the window without blocking
        time.sleep(0.01)  # Sleep for a short time to allow other threads to execute


def audio_callback_random_decoder(indata, frames, time, status):
    """
    Callback to process audio input and map it to latent space.
    """
    global latest_image

    if status:
        print(status)
    
    # Flatten audio input and normalize
    audio_data = indata[:, 0]  # Single-channel (mono)
    audio_energy = np.sqrt(np.mean(audio_data**2))  # Compute RMS energy

    if audio_energy < NOISE_THRESHOLD:
        # Ignore this chunk if it's below the noise threshold
        return

    # Compute FFT (Fast Fourier Transform)
    fft_data = np.abs(scipy.fftpack.fft(audio_data, n=1024))[:512]  # Keep first 512 coefficients

    # Map FFT data to latent space
    latent_vector = np.dot(fft_data, random_projection[:512, :])

    # Generate an image from the latent vector using the random decoder
    latent_tensor = torch.from_numpy(latent_vector).float().unsqueeze(0)  # Convert to tensor and add batch dimension
    generated_image = random_decoder(latent_tensor)

    # Convert generated image to NumPy array for OpenCV
    generated_image = generated_image.squeeze().detach().cpu().numpy()
    generated_image = (generated_image + 1) * 127.5  # Convert from [-1, 1] to [0, 255]
    generated_image = generated_image.astype(np.uint8)  # Ensure integer type

    # Resize image to fit window size
    generated_image = np.transpose(generated_image, (1, 2, 0))  # Convert from CxHxW to HxWxC
    generated_image = cv2.resize(generated_image, (1280, 720))

    # Update the latest_image to be displayed
    latest_image = generated_image


# def audio_callback_pretrained_decoder(indata, frames, time, status):
#     """
#     Callback to process audio input and map it to latent space.
#     """
#     if status:
#         print(status)
#     # Flatten audio input and normalize
#     audio_data = indata[:, 0]  # Single-channel (mono)
#     audio_energy = np.sqrt(np.mean(audio_data**2))  # Compute RMS energy

#     if audio_energy < NOISE_THRESHOLD:
#         # Ignore this chunk if it's below the noise threshold
#         return

#     # Compute FFT (Fast Fourier Transform)
#     fft_data = np.abs(scipy.fftpack.fft(audio_data, n=1024))[:512]  # Keep first 512 coefficients

#     # Map FFT data to latent space
#     latent_vector = np.dot(fft_data, random_projection[:512, :])

#     # Generate an image from the latent vector using the StyleGAN model
#     latent_tensor = torch.from_numpy(latent_vector).float().unsqueeze(0)  # Convert to tensor and add batch dimension
#     generated_image = model_loader.generate(latent_tensor)

#     # Convert generated image to NumPy array for OpenCV
#     generated_image = generated_image.squeeze().detach().cpu().numpy()
#     generated_image = (generated_image * 255).astype(np.uint8)  # Normalize to [0, 255]

#     # Resize image to fit window size
#     generated_image = cv2.resize(generated_image, (512, 512))

#     # Display the image in real-time
#     cv2.imshow("Generated Image", generated_image)
# # Start the display thread
# display_thread = threading.Thread(target=display_image, daemon=True)
# display_thread.start()


def main():
    audio_device_index = 51
    SAMPLERATE = 44100  # Sampling rate of device
    
    #Decoder to generate the image
    callback_function = audio_callback_random_decoder

    # Start audio stream
    with sd.InputStream(
        device=audio_device_index,
        callback=callback_function,
        channels=1,
        samplerate=SAMPLERATE,
        blocksize=int(DURATION * SAMPLERATE)
    ):
        
        print("Recording... Press Ctrl+C to stop.")
        try:
            while True:
                display_image()
                time.sleep(0.01)  # Small sleep to keep the main thread responsive
        except KeyboardInterrupt:
            print("\nStopped.")
            cv2.destroyAllWindows()


if __name__ == '__main__':
    pass
    # main()