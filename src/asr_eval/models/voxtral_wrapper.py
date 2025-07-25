from typing import Literal, Any, List, Optional
import httpx
import soundfile as sf
import tempfile
import os
from typing_extensions import override
import io
from mistral_common.protocol.instruct.messages import TextChunk, AudioChunk, UserMessage
from mistral_common.audio import Audio
from openai import OpenAI

from .base import ASREvalWrapper, TimedText
from ..utils.types import FLOATS

class VoxtralWrapper(ASREvalWrapper):
    def __init__(
        self,
        api: str,
        model_name: str = 'mistralai/Voxtral-Small-24B-2507',
        lang: Literal['ru', 'en'] = 'ru',
        temperature: float = 0.0,
        top_p: float = 0.95,
    ):
        self.model_name = model_name
        self.lang = lang
        self.api = api
        self.temperature = temperature
        self.top_p = top_p
        
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=self.api,
        )

    def is_voxtral_healthy(self, timeout_sec: int = 5) -> bool:
        try:
            response = httpx.get(f"{self.api.replace('/v1', '')}/health", timeout=timeout_sec)
            return response.status_code == 200
        except Exception:
            return False

    def _waveform_to_audio_chunk(self, waveform: FLOATS) -> AudioChunk:
        """Convert waveform to AudioChunk using temporary file."""
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, waveform, 16000, format='WAV')  # type: ignore
        
        try:
            # Load audio and create chunk
            audio = Audio.from_file(temp_path, strict=False)
            return AudioChunk.from_audio(audio)
        finally:
            os.unlink(temp_path)

    
    @override
    def __call__(self, waveforms: list[FLOATS], **kwargs: Any) -> list[str]:
        return [self.transcribe(waveform, **kwargs)[0].text for waveform in waveforms]


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
            sf.write(wav_buffer, waveform, 16000, format='WAV')
            wav_buffer.seek(0)
            files = {'file': ('audio.wav', wav_buffer, 'audio/wav')}
            data = {
                'model': self.model_name,
                'language': self.lang,
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
                else:
                    raise RuntimeError(f"HTTP error {response.status_code} from Voxtral API")

            
            audio_chunk = self._waveform_to_audio_chunk(waveform)

            system_prompt = f'Recognize the audio on {self.lang} language.'
            
            base_prompt = kwargs.get('prompt')
            prev_transcription = kwargs.get('prev_transcription')

            if prev_transcription:
                if base_prompt:
                    full_prompt = f"{base_prompt}. Предыдущая транскрипция, которую ты можешь использовать для улучшения транскрипции: {prev_transcription}."
                else:
                    full_prompt = f"Предыдущая транскрипция, которую ты можешь использовать для улучшения транскрипции: {prev_transcription}"
            else:
                full_prompt = base_prompt

            full_prompt = f"{system_prompt}.Сгенерированная транскрипция аудио из аудиофайла: {text}.Используй ее для улучшения транскрипции. {full_prompt}"
            
            text_chunk = TextChunk(text=full_prompt)
            
            user_msg = UserMessage(content=[audio_chunk, text_chunk]).to_openai()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[user_msg],  # type: ignore
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            text = response.choices[0].message.content or ""
            return [TimedText(0, len(waveform) / 16000, text)]

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Error processing audio with Voxtral: {str(e)}")