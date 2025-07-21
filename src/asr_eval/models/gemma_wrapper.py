import os
import torch
import numpy as np
import soundfile as sf
from tempfile import NamedTemporaryFile
from typing import List, override
from transformers import AutoProcessor, AutoModelForImageTextToText
from src.asr_eval.models.base import ASREvalWrapper
from src.asr_eval.utils.types import FLOATS

class Gemma3nSTTWrapper(ASREvalWrapper):
    def __init__(
        self,
        token: str,
        model_name: str = "google/gemma-3n-E4B-it",
        torch_dtype: torch.dtype = torch.bfloat16,
        task_prompt: str = "Транскрибируй текст из аудио на русском языке. Пиши только транскрибацию и ничего больше.",
    ):
        """

        Args:
            token (str, optional): HF-token. Defaults to "".
            model_name (str, optional): Name of model. Defaults to "google/gemma-3n-E4B-it".
            torch_dtype (torch.dtype, optional): torch_dtype. Defaults to torch.bfloat16 (recommended).
            task_prompt (str, optional): Prompt. Defaults to "Транскрибируй этот аудио-файл. Пиши только транскрибацию и больше ничего.".
        """
        
        self.token = token
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype
        self.task_prompt = task_prompt
        
        self.model = None
        self.processor = None

    def _waveform_to_temp_wav(self, waveform: FLOATS, sampling_rate: int = 16000) -> str:
        """Convert waveform to temporary WAV-file"""
        waveform = np.asarray(waveform)
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        
        # Нормализация
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak
        
        with NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write( # type: ignore
                tmp_file.name,
                waveform,
                sampling_rate,
                subtype='FLOAT'
            )
            return tmp_file.name
    
    def _maybe_instantiate(self):
        if self.model is None: # type: ignore
            self.processor = AutoProcessor.from_pretrained( # type: ignore
                self.model_name,
                token=self.token
            )
            self.model = AutoModelForImageTextToText.from_pretrained( # type: ignore
                self.model_name,
                torch_dtype=self.torch_dtype,
                token=self.token
            )

    @override
    def __call__(self, waveforms: List[FLOATS], sampling_rate: int = 16000) -> List[str]:
        self._maybe_instantiate()
        transcripts = []
        
        for waveform in waveforms:
            # 1. Подготовка аудио (как в Whisper)
            waveform = np.asarray(waveform)
            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)
            
            # 2. Конвертация в WAV (адаптер для Gemma)
            wav_path = self._waveform_to_temp_wav(waveform, sampling_rate)
            
            # 3. Формируем сообщение для Gemma
            messages = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": wav_path},
                    {"type": "text", "text": self.task_prompt},
                ]
            }]
            
            # 4. Обработка через модель
            inputs = self.processor.apply_chat_template(  # type: ignore
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device, dtype=self.torch_dtype) # type: ignore
            
            with torch.no_grad():
                outputs = self.model.generate( # type: ignore
                    **inputs,
                    do_sample=False,
                    temperature=0.1,
                )
            
            # 5. Декодирование и очистка
            transcript = self.processor.batch_decode(  # type: ignore
                outputs,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )[0]
            
            # 6. Удаление временного файла
            os.unlink(wav_path) 
            
            transcripts.append(transcript) # type: ignore
        
        return transcripts # type: ignore