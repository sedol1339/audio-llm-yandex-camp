from typing import Literal, Any
import httpx
import io
import soundfile as sf
from typing_extensions import override

from .base import ASREvalWrapper, TimedText
from ..utils.types import FLOATS

class VoxtralmWrapper(ASREvalWrapper):
    def __init__(
        self,
        api: str = "http://84.201.157.110:8000/v1",
        model_name: str = 'mistralai/Voxtral-Small-24B-2507',
        lang: Literal['ru', 'en'] = 'ru',
    ):
        self.model_name = model_name
        self.lang = lang
        self.api = api

    def is_voxtral_healthy(self, timeout_sec: int = 5) -> bool:
        try:
            response = httpx.get(f"{self.api.replace('/v1', '')}/health", timeout=timeout_sec)
            return response.status_code == 200
        except Exception:
            return False

    @override
    def transcribe(self, waveform: FLOATS, **kwargs: Any) -> list[TimedText]:
        if self.is_voxtral_healthy():
            try:
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, waveform, 16000, format='WAV')  # type: ignore
                wav_buffer.seek(0)
                files = {'file': ('audio.wav', wav_buffer, 'audio/wav')}

                data = {
                    'model': self.model_name,
                    'language': self.lang,
                    'prompt': kwargs.get('prompt', ''),
                    'response_format': 'json',
                    'temperature': kwargs.get('temperature', 0.5),
                }

                with httpx.Client() as client:
                    response = client.post(
                        f"{self.api}/audio/transcriptions",
                        data=data,
                        files=files,
                        timeout=30
                    )
                    if response.status_code == 200:
                        json_response = response.json()
                        text = json_response.get('text', '')
                        return [TimedText(0, len(waveform) / 16000, text)]
                    else:
                        return [TimedText(0, len(waveform) / 16000, f"Error: HTTP {response.status_code}")]

            except Exception as e:
                return [TimedText(0, len(waveform) / 16000, f"Error processing audio: {str(e)}")]
        else:
            return [TimedText(0, len(waveform) / 16000, "Model doesn't work")]