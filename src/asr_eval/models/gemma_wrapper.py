# type: ignore

import os
import re
import torch
import numpy as np
import soundfile as sf
from tempfile import NamedTemporaryFile
from typing import List, override, Literal, Optional
from transformers import AutoProcessor, AutoModelForImageTextToText
from src.asr_eval.models.base import ASREvalWrapper
from src.asr_eval.utils.types import FLOATS
from ..segments.chunking import chunk_audio  
from ..segments.segment import AudioSegment  

# def extract_transcript(raw_text: str) -> str:
#     """Улучшенное извлечение транскрипции с обработкой ошибок"""
#     # Паттерны для обнаружения отказа модели
#     rejection_patterns = [
#         r"я не могу",
#         r"не является текстом",
#         r"не удалось",
#         r"пусто",
#         r"неразборчиво",
#         r"не понимаю"
#     ]
    
#     for pattern in rejection_patterns:
#         if re.search(pattern, raw_text, re.IGNORECASE):
#             return ""
    
#     # Упрощенная логика извлечения
#     lines = raw_text.split('\n')
#     cleaned_lines = []
    
#     for line in lines:
#         line = line.strip()
#         # Пропускаем служебные строки
#         if not line or line.startswith("<") or "http" in line:
#             continue
#         # Удаляем метки типа [музыка] или [шум]
#         if line.startswith("[") and line.endswith("]"):
#             continue
#         cleaned_lines.append(line)
    
#     # Объединяем строки, убирая повторы
#     transcript = " ".join(cleaned_lines)
    
#     # Удаляем повторяющиеся слова (артефакты модели)
#     words = transcript.split()
#     deduped_words = [words[i] for i in range(len(words)) 
#                      if i == 0 or words[i] != words[i-1]]
    
#     return " ".join(deduped_words).strip()

def extract_transcript(raw_text):  
    rejection_patterns = [
        r"я не могу",
        r"не является текстом",
        r"не удалось",
        r"пусто",
        r"неразборчиво",
        r"не понимаю"
    ]
    
    for pattern in rejection_patterns:
        if re.search(pattern, raw_text, re.IGNORECASE):
            return ""
    
    # Разделяем текст по строкам
    lines = raw_text.split('\n')  
    
    # Флаг для определения, когда мы достигли нужной части
    found_model = False
    transcript = []
    
    for line in lines:  
        line = line.strip()  
        if not line:
            continue  # Пропускаем пустые строки
        
        if line.lower().startswith('model'):  
            found_model = True
            # Берем текст после слова 'model' в этой строке
            model_line = line[5:].strip()  # 5 - длина 'model'  
            if model_line:
                transcript.append(model_line)  
            continue
        
        if found_model:
            # Если строка не содержит тегов (простая проверка)
            if '<' not in line and '>' not in line:
                transcript.append(line)  
    
    # Собираем результат
    return ' '.join(transcript).strip()  


class Gemma3nSTTWrapper(ASREvalWrapper):
    def __init__(
        self,
        token: str,
        model_name: str = "google/gemma-3n-E4B-it",
        torch_dtype: torch.dtype = torch.bfloat16,
        task_prompt: str = "Транскрибируй только русскую речь из аудио. Если нет речи - верни пустую строку. Не комментируй.",
        lang: Literal['ru', 'en'] = 'ru',
        segment_length: float = 30.0,
        segment_shift: float = 15.0,  # Увеличено перекрытие
        min_chunk_duration: float = 1.0  # Новый параметр
    ):
        self.token = token
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype
        self.task_prompt = task_prompt
        self.lang = lang
        self.segment_length = segment_length
        self.segment_shift = segment_shift
        self.min_chunk_duration = min_chunk_duration  # Минимальная длительность сегмента
        
        self.max_new_tokens = 50000
        self.model = None
        self.processor = None
        self.retry_count = 2  # Количество попыток при ошибке

    def _waveform_to_temp_wav(self, waveform: FLOATS, sampling_rate: int = 16000) -> str:
        waveform = np.asarray(waveform, dtype=np.float32)
        # Нормализация с защитой от тишины
        peak = np.max(np.abs(waveform))
        if peak > 0.01:  # Порог для избежания усиления шума
            waveform /= peak
        else:  # Если сигнал слишком тихий
            waveform = np.zeros_like(waveform)
        
        with NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, waveform, sampling_rate, subtype='FLOAT')
            return tmp_file.name
    
    def _maybe_instantiate(self):
        if self.model is None:  
            self.processor = AutoProcessor.from_pretrained(  
                self.model_name,
                token=self.token,
                language='Russian' if self.lang == 'ru' else 'English',
                do_sample=False,
                temperature=0.7  # Слегка увеличена случайность
            )
            self.model = AutoModelForImageTextToText.from_pretrained(  
                self.model_name,
                torch_dtype=self.torch_dtype,
                token=self.token
            ).to(self.device)
            self.model.eval()

    def _transcribe_chunk(self, waveform: FLOATS, sampling_rate: int) -> str:
        """Транскрипция с повторными попытками и проверкой качества"""
        # Пропускаем слишком короткие сегменты
        if len(waveform) / sampling_rate < self.min_chunk_duration:
            return ""
        
        best_transcript = ""
        for attempt in range(self.retry_count):
            try:
                wav_path = self._waveform_to_temp_wav(waveform, sampling_rate)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": wav_path},
                        {"type": "text", "text": self.task_prompt},
                    ]
                }]
                
                inputs = self.processor.apply_chat_template(  
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(self.model.device, dtype=self.torch_dtype)  
                
                with torch.no_grad():
                    outputs = self.model.generate(  
                        **inputs,
                        max_new_tokens=min(self.max_new_tokens, 100 + int(len(waveform)/sampling_rate * 10)),
                        do_sample=True,  # Разрешена случайность
                        num_beams=2,  # Улучшенный поиск
                        temperature=0.7,
                        repetition_penalty=1.2  # Снижение повторов
                    )
                
                raw_transcript = self.processor.batch_decode(  
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
                
                os.unlink(wav_path)
                transcript = extract_transcript(raw_transcript)
                
                # Выбираем лучшую транскрипцию по длине
                if len(transcript) > len(best_transcript):
                    best_transcript = transcript
                    
            except Exception as e:
                print(f"Transcription error: {str(e)}")
        
        return best_transcript

    def _merge_transcripts(self, transcripts: List[str]) -> str:
        """Улучшенное объединение с фильтрацией пустых результатов"""
        non_empty = [t for t in transcripts if t.strip()]
        if not non_empty:
            return ""
        
        merged = non_empty[0]
        for next_text in non_empty[1:]:
            merged_words = merged.split()
            next_words = next_text.split()
            
            # Ищем перекрытие 3-5 слов
            overlap_found = False
            for overlap_size in range(min(5, len(merged_words)), 2, -1):
                if len(merged_words) < overlap_size or len(next_words) < overlap_size:
                    continue
                    
                if merged_words[-overlap_size:] == next_words[:overlap_size]:
                    merged += " " + " ".join(next_words[overlap_size:])
                    overlap_found = True
                    break
            
            if not overlap_found:
                merged += " " + next_text
                
        return merged

    @override
    def __call__(self, waveforms: List[FLOATS], sampling_rate: int = 16000) -> List[str]:
        self._maybe_instantiate()
        transcripts = []
        
        for waveform in waveforms:
            waveform = np.asarray(waveform, dtype=np.float32)
            duration = len(waveform) / sampling_rate
            
            # Для очень коротких аудио добавляем паддинг
            if duration < 0.5:
                transcripts.append("")
                continue
            elif duration < 2.0:  # Короткие аудио обрабатываем с паддингом
                padded = np.zeros(int(2.0 * sampling_rate))
                start = (len(padded) - len(waveform)) // 2
                padded[start:start+len(waveform)] = waveform
                waveform = padded
                duration = 2.0
                
            if duration <= self.segment_length:
                transcripts.append(self._transcribe_chunk(waveform, sampling_rate))
                continue
                
            segments = chunk_audio(  
                length=duration,
                segment_length=self.segment_length,
                segment_shift=self.segment_shift,
                last_chunk_mode='same_length'
            )
            
            chunk_transcripts = []
            for seg in segments:
                start_idx = seg.start_pos(sampling_rate)
                end_idx = seg.end_pos(sampling_rate)
                audio_chunk = waveform[start_idx:end_idx]
                transcript = self._transcribe_chunk(audio_chunk, sampling_rate)
                
                # Фильтрация артефактов
                if transcript and len(transcript.split()) < 3:  # Отбрасываем слишком короткие
                    if any(word in transcript.lower() for word in ["я", "а", "и", "о", "у"]):
                        transcript = ""
                
                chunk_transcripts.append(transcript)
            
            transcripts.append(self._merge_transcripts(chunk_transcripts))
        
        return transcripts