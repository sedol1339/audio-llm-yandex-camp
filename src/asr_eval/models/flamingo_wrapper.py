from typing import Any, List

import numpy as np
import torch
from huggingface_hub import snapshot_download

import llava

from .base import FLOATS, ASREvalWrapper, TimedText, audio_as_file

# Define the data structures for clarity, as in your example


# ---------------------------------
# Flamingo Wrapper Implementation
# ---------------------------------
class FlamingoWrapper(ASREvalWrapper):
    """
    A wrapper for the nvidia/audio-flamingo-3 model that conforms to the ASREvalWrapper interface.
    """

    def __init__(
        self, model_repo_id: str = "nvidia/audio-flamingo-3", device: str = "cuda"
    ):
        """
        Initializes the wrapper and loads the Flamingo model into memory.

        Args:
            model_repo_id (str): The repository ID on Hugging Face Hub.
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        print("Initializing FlamingoWrapper and loading model...")
        # Check for GPU availability
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU.")
            device = "cpu"

        self.device = device
        self.sampling_rate = 16_000

        # Load the model from Hugging Face Hub
        model_base_path = snapshot_download(repo_id=model_repo_id)
        self.model = llava.load(model_base_path, model_base=None)
        self.model = self.model.to(self.device)

        # Store the default generation configuration
        self.generation_config = self.model.default_generation_config
        print("Model loaded successfully.")

    def transcribe(self, waveforms: list[FLOATS], **kwargs: Any) -> List[TimedText]:
        """
        Transcribes or analyzes an audio waveform using the Flamingo model.

        Args:
            waveform (np.ndarray): A 1D numpy array of audio data (float).
                                   The expected sampling rate is 16,000 Hz.
            **kwargs: Must contain a 'prompt' (str) for the model.

        Returns:
            A list containing a single TimedText object with the model's full response.
        """
        # Get the required text prompt from keyword arguments
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError("A 'prompt' must be provided in kwargs.")

        result: list[TimedText] = []
        for waveform in waveforms:
            try:
                # Create a llava.Sound object directly from the waveform and sampling rate
                # This is the key change to work with raw audio data instead of a file path
                audio_path = audio_as_file(waveform)
                sound = llava.Sound(audio_path)

                # Construct the full prompt for the model
                full_prompt = f"<sound>\n{prompt}"

                # Generate content using the model
                response_text = self.model.generate_content(
                    [sound, full_prompt], generation_config=self.generation_config
                )

                # As Flamingo does not provide timestamps, we create one segment for the whole audio
                duration_seconds = len(waveform) / self.sampling_rate
                result.append(TimedText(text=response_text, start_time=0, end_time=0))

            except Exception as e:
                print(f"An error occurred during transcription: {e}")
        return result

    def transcribe_from_file(self, files: list[str], **kwargs: Any) -> List[TimedText]:
        """
        Transcribes or analyzes an audio waveform using the Flamingo model.

        Args:
            waveform (np.ndarray): A 1D numpy array of audio data (float).
                                   The expected sampling rate is 16,000 Hz.
            **kwargs: Must contain a 'prompt' (str) for the model.

        Returns:
            A list containing a single TimedText object with the model's full response.
        """
        # Get the required text prompt from keyword arguments
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError("A 'prompt' must be provided in kwargs.")

        result: list[TimedText] = []
        for file in files:
            try:
                # Create a llava.Sound object directly from the waveform and sampling rate
                # This is the key change to work with raw audio data instead of a file path
                sound = llava.Sound(file)

                # Construct the full prompt for the model
                full_prompt = f"\n{prompt}"

                # Generate content using the model
                response_text = self.model.generate_content(
                    [sound, full_prompt], generation_config=self.generation_config
                )

                # As Flamingo does not provide timestamps, we create one segment for the whole audio
                result.append(TimedText(text=response_text, start_time=0, end_time=0))

            except Exception as e:
                print(f"An error occurred during transcription: {e}")
        return result


if __name__ == "__main__":
    print("--- Running FlamingoWrapper Example ---")

    # 1. Initialize the wrapper. The model will be downloaded and loaded here.
    #    This might take some time on the first run.
    try:
        asr_wrapper = FlamingoWrapper(device="cuda")

        # 2. Create a dummy audio waveform for demonstration.
        #    In a real scenario, you would load your audio file here, for example using
        #    a library like 'librosa' or 'soundfile' and ensure it's 16kHz.
        duration_in_seconds = 5
        sampling_rate = 16000
        dummy_waveform = np.random.uniform(
            low=-0.5, high=0.5, size=(duration_in_seconds * sampling_rate,)
        ).astype(np.float32)
        print(f"\nCreated a dummy audio waveform of {duration_in_seconds} seconds.")

        # 3. Define the prompt for the model.
        #    This is the question you want to ask about the audio.
        my_prompt = "Transcribe what is said in the audio."
        print(f"Using prompt: '{my_prompt}'")

        # 4. Call the 'transcribe' method.
        #    Pass the waveform and the prompt.
        transcription_results = asr_wrapper.transcribe(dummy_waveform, prompt=my_prompt)

        # 5. Print the results.
        if transcription_results:
            result = transcription_results[0]
            print("\n--- Transcription Result ---")
            print(f"Text: {result.text}")
            print(f"Start: {result.start:.2f}s, End: {result.end:.2f}s")
            print("--------------------------")
        else:
            print("\nTranscription failed.")

    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")
