import os
import numpy as np
import soundfile as sf
from tempfile import NamedTemporaryFile
from abc import ABC, abstractmethod
from typing import List, Any

# Yandex SpeechKit imports
from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType
from ..utils.formatting import waveform_to_temp_wav
from ..utils.types import FLOATS
from .base import ASREvalWrapper, TimedText


SAMPLING_RATE = 16000

class YandexSpeechKitWrapper(ASREvalWrapper):
    """
    A wrapper for Yandex SpeechKit that conforms to the ASREvalWrapper interface.
    """
    def __init__(
        self,
        api_key: str = os.getenv('API_KEY'),
        model: str = 'general',
        language: str = 'ru-RU',
        audio_processing: AudioProcessingType = AudioProcessingType.Full
    ):
        """
        Initializes the wrapper, configures credentials, and sets up the recognition model.
        
        :param api_key: Your Yandex Cloud API key.
        :param model: Recognition model name (e.g., 'general', 'general:rc').
        :param language: Language code (e.g., 'ru-RU').
        :param audio_processing: Type of audio processing.
        """
        if not api_key:
            raise ValueError("Yandex API key is required.")

        # Configure credentials once during initialization
        configure_credentials(yandex_credentials=creds.YandexCredentials(api_key=api_key))

        # Setup the recognition model
        self.model = model_repository.recognition_model()
        self.model.model = model
        self.model.language = language
        self.model.audio_processing_type = audio_processing

    @override
    def transcribe(self, waveform: FLOATS, **kwargs: Any) -> list[TimedText]:
        return [TimedText(0, len(waveform) / SAMPLING_RATE, self.__call__([waveform]))]

    def __call__(self, waveforms: List[FLOATS]) -> List[str]:
        """
        Transcribes a list of audio waveforms.

        :param waveforms: A list of waveforms, where each waveform is a list of floats.
        :return: A list of transcribed texts.
        """
        transcriptions = []
        print(waveform)
        for waveform in waveforms:
            temp_audio_path = None
            try:
                # Convert waveform to a temporary WAV file
                temp_audio_path = waveform_to_temp_wav(waveform)
                
                # Transcribe the file
                result = self.model.transcribe_file(temp_audio_path)
                
                # We assume single-channel audio, so we take the first result
                if result:
                    transcriptions.append(result[0].normalized_text)
                else:
                    transcriptions.append("") # Append empty string if no result
            
            finally:
                # Ensure the temporary file is always deleted
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
        
        return transcriptions
