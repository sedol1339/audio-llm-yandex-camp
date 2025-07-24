from typing import Literal, Any, List, Optional
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
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.lang = lang
        self.api = api
        self.temperature = temperature

    def is_voxtral_healthy(self, timeout_sec: int = 5) -> bool:
        try:
            response = httpx.get(f"{self.api.replace('/v1', '')}/health", timeout=timeout_sec)
            return response.status_code == 200
        except Exception:
            return False

    @override
    def transcribe(self, waveform: FLOATS, **kwargs: Any) -> list[TimedText]:
        if not self.is_voxtral_healthy():
            raise RuntimeError("Voxtral model is not healthy or not responding")
        
        if len(waveform) == 0:
            return [TimedText(0, 0, "")]
        
        if len(waveform) < 100:
            print(f"Warning: Very short waveform ({len(waveform)} samples), returning empty transcription")
            return [TimedText(0, len(waveform) / 16000, "")]
        
        try:
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, waveform, 16000, format='WAV')  # type: ignore
            wav_buffer.seek(0)
            files = {'file': ('audio.wav', wav_buffer, 'audio/wav')}

            base_prompt = kwargs.get('prompt', '')
            prev_transcription = kwargs.get('prev_transcription')
            
            if prev_transcription:
                if base_prompt:
                    full_prompt = f"{base_prompt} Previous transcription: {prev_transcription}"
                else:
                    full_prompt = f"Previous transcription: {prev_transcription}"
            else:
                full_prompt = base_prompt

            data = {
                'model': self.model_name,
                'language': self.lang,
                'prompt': full_prompt,
                'response_format': 'json',
                'temperature': self.temperature,
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
                    raise RuntimeError(f"HTTP error {response.status_code} from Voxtral API")

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Error processing audio with Voxtral: {str(e)}")

    def transcribe_chat(self, waveform: FLOATS, **kwargs: Any) -> list[TimedText]: # THIS IS TEST FUNCTION

        if not self.is_voxtral_healthy():
            raise RuntimeError("Voxtral model is not healthy or not responding")
        
        try:
            basic_transcription = self.transcribe(waveform, **kwargs)
            if not basic_transcription:
                return [TimedText(0, len(waveform) / 16000, "")]
            
            basic_text = basic_transcription[0].text
            
            base_prompt = kwargs.get('prompt', f'Транскрибируй аудио на {self.lang} языке.')
            prev_transcription = kwargs.get('prev_transcription')
            
            user_content = f"{base_prompt}\n\nТранскрипция: {basic_text}"
            if prev_transcription:
                user_content = f"{base_prompt}\n\nПредыдущая транскрипция: {prev_transcription}\n\nТекущая транскрипция: {basic_text}"
            
            messages = [
                {
                    "role": "system",
                    "content": "Ты - эксперт по транскрипции аудио. Твоя задача - улучшить и отредактировать предоставленную транскрипцию согласно инструкциям. Ты должен вернуть ТОЛЬКО транскрипцию."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]

            data = {
                'model': self.model_name,
                'messages': messages,
                'temperature': self.temperature,
                'max_tokens': kwargs.get('max_tokens', 1000),
                'stream': False
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.api}/chat/completions",
                    json=data,
                    timeout=30
                )
                if response.status_code == 200:
                    json_response = response.json()
                    improved_text = json_response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    return [TimedText(0, len(waveform) / 16000, improved_text)]
                else:
                    error_detail = response.text if response.text else "No error details"
                    raise RuntimeError(f"HTTP error {response.status_code} from Voxtral chat API: {error_detail}")

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Error processing audio with Voxtral chat: {str(e)}")
