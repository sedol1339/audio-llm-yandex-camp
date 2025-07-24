from typing import Literal
from typing_extensions import override  # Для Python < 3.12
import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor # type: ignore
from src.asr_eval.models.base import ASREvalWrapper
from src.asr_eval.utils.types import FLOATS

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
            'return_timestamps': True,
            'max_new_tokens': 10_000,
            'condition_on_prev_tokens': False,
        }

    def _maybe_instantiate(self):
        if self.model is None:
            self.processor = WhisperProcessor.from_pretrained( # type: ignore
                self.model_name,
                language='Russian' if self.lang == 'ru' else 'English'
            )
            
            self.model = WhisperForConditionalGeneration.from_pretrained( # type: ignore
                self.model_name,
                attn_implementation="sdpa",
                torch_dtype=torch.float32
            ).to(self.device) # type: ignore

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
            inputs = self.processor( # type: ignore
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding="longest",  # Добавлено для поддержки разных длин
                truncation=False,
            )
            
            # 3. Перенос на нужное устройство с правильным типом
            input_features = inputs.get("input_features") # type: ignore
            if input_features is None:
                input_features = inputs.get("input_values")  # type: ignore # Альтернативное имя для разных версий
                
            if input_features is None:
                raise ValueError("Processor не вернул ни input_features, ни input_values")
                
            input_features = input_features.to(device=self.device, dtype=self.model.dtype) # type: ignore
            
            # 4. Генерация текста
            with torch.no_grad():
                outputs = self.model.generate( # type: ignore
                    input_features=input_features,  # type: ignore
                    **self.generate_kwargs # type: ignore
                )
            
            text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0] # type: ignore
            texts.append(text) # type: ignore
        
        return texts # type: ignore