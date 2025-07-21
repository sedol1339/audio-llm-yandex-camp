from typing import Literal, cast
from typing_extensions import override  # Для Python < 3.12
import torch
import numpy as np
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
from .base import ASREvalWrapper
from ..utils.types import FLOATS

class WhisperLongformWrapper(ASREvalWrapper):
    def __init__(
        self,
        model_name: str = 'openai/whisper-large-v3',
        lang: Literal['ru', 'en'] = 'ru',
        condition_on_prev_tokens: bool = False,
    ):
        self.model_name = model_name
        self.lang = lang
        self.condition_on_prev_tokens = condition_on_prev_tokens
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.processor = None
        self.generate_kwargs = {
            'language': self.lang,
            'task': 'transcribe',
            'temperature': 0,
            'do_sample': False,
            'return_timestamps': True
        }

    def _maybe_instantiate(self):
        if self.model is None:
            self.processor = WhisperProcessor.from_pretrained(
                self.model_name,
                language='Russian' if self.lang == 'ru' else 'English'
            )
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                attn_implementation="sdpa",
                torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32
            ).to(self.device)

    @override
    def __call__(self, waveforms: list[FLOATS]) -> list[str]:
        self._maybe_instantiate()
        texts = []
        
        for waveform in waveforms:
            # 1. Конвертация и нормализация аудио
            waveform = np.asarray(waveform)
            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)
            waveform = waveform / np.max(np.abs(waveform))  # Нормализация [-1, 1]
            
            # 2. Подготовка входных данных (исправленная версия)
            inputs = self.processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding="longest"  # Добавлено для поддержки разных длин
            )
            
            # 3. Перенос на нужное устройство с правильным типом
            input_features = inputs.get("input_features")
            if input_features is None:
                input_features = inputs.get("input_values")  # Альтернативное имя для разных версий
                
            if input_features is None:
                raise ValueError("Processor не вернул ни input_features, ни input_values")
                
            input_features = input_features.to(device=self.device, dtype=self.model.dtype)
            
            # 4. Генерация текста
            with torch.no_grad():
                outputs = self.model.generate(
                    input_features=input_features,
                    **self.generate_kwargs
                )
            
            text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            texts.append(text)
        
        return texts