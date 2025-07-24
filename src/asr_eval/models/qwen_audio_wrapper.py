import torch
from typing import override, List, Any
import numpy as np

# Критически важные импорты
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# Ваши базовые классы и функции для чанкинга
from .base import ASREvalWrapper, TimedText
from ..utils.types import FLOATS
from ..segments.chunking import chunk_audio

# Константа для частоты дискретизации, ожидаемой моделью
SAMPLING_RATE = 16000

class QwenAudioWrapper(ASREvalWrapper):
    """
    Финальная версия обертки для Qwen-Audio с поддержкой длинных аудиофайлов
    через механизм нарезки на чанки.
    """
    def __init__(
        self,
        model_name: str = 'Qwen/Qwen2-Audio-7B-Instruct',
        # transcription_prompt: str = "Дословно и высокоточно транскрибируй на русский язык следующую аудиозапись на русском языке, не добавляя комментари, не исправляя ошибки говорящего и не теряя слова. Твой ответ должен содержать только итоговый текст транскрибации на русском языке без потери слов и без добавления новых слов.",
        transcription_prompt: str = "Verbatim and accurately transcribe the following audio recording into Russian without adding comments, without correcting the speaker's mistakes, without translating the text into other languages, without adding new words, and without losing words. Your answer must contain only the final transcription text in Russian, without losing words or adding new words.",
        # Параметры для нарезки на чанки
        segment_length_s: float = 30.0, # Длина каждого чанка в секундах
        segment_shift_s: float = 10.0,  # Шаг (перекрытие) между чанками в секундах
    ):
        self.model_name = model_name
        self.transcription_prompt = transcription_prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Параметры чанкинга
        self.segment_length_s = segment_length_s
        self.segment_shift_s = segment_shift_s

        self.model: Qwen2AudioForConditionalGeneration | None = None
        self.processor: AutoProcessor | None = None

    def _maybe_instantiate(self):
        # Эта часть остается без изменений
        if self.processor is None or self.model is None:
            print(f"Loading Qwen-Audio model '{self.model_name}' and processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_name, device_map="auto", trust_remote_code=True
            )
            print("Model and processor loaded successfully.")

    def _transcribe_single_chunk(self, waveform_chunk: FLOATS) -> str:
        """
        Вспомогательная функция для транскрибации одного короткого фрагмента аудио.
        """
        # Эта логика была перенесена из старого __call__
        conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": self.transcription_prompt},
                {"type": "audio"},
            ]}
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(
            text=[prompt], audios=[waveform_chunk], return_tensors="pt", padding=True
        ).to(self.device)

        generate_ids = self.model.generate(
            **inputs, max_new_tokens=1024, do_sample=False, temperature=0.0
        )

        input_token_len = inputs.input_ids.shape[1]
        output_ids = generate_ids[:, input_token_len:]
        
        result_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return result_text.strip()

    @staticmethod
    def _stitch_transcripts(transcripts: List[str]) -> str:
        """
        Сшивает список перекрывающихся транскрипций в одну.
        """
        def find_overlap(text1: str, text2: str) -> int:
            """Находит длину максимального суффикса text1, который является префиксом text2."""
            max_overlap = 0
            # Ищем перекрытие, сравнивая конец первого текста с началом второго
            for i in range(min(len(text1), len(text2)), 0, -1):
                if text1.endswith(text2[:i]):
                    max_overlap = i
                    break
            # Простой эвристический фильтр, чтобы избежать случайных совпадений из одного слова
            if max_overlap > 0 and ' ' not in text2[:max_overlap]:
                return 0 # Игнорируем совпадения из одного слова
            return max_overlap

        full_text = transcripts[0]
        for i in range(1, len(transcripts)):
            overlap_len = find_overlap(full_text, transcripts[i])
            full_text += transcripts[i][overlap_len:]
        return full_text
    

    @override
    def transcribe(self, waveform: FLOATS, **kwargs: Any) -> list[TimedText]:
        return [TimedText(0, len(waveform) / SAMPLING_RATE, self.__call__([waveform]))]

    @override
    @torch.inference_mode()
    def __call__(self, waveforms: list[FLOATS]) -> list[str]:
        self._maybe_instantiate()
        if self.model is None or self.processor is None:
            raise RuntimeError("Model or processor not instantiated.")

        final_transcriptions = []
        for waveform in waveforms:
            audio_duration_s = len(waveform) / SAMPLING_RATE
            
            # Если аудио короткое, обрабатываем его целиком без нарезки
            if audio_duration_s <= self.segment_length_s:
                print("Processing short audio directly...")
                transcription = self._transcribe_single_chunk(waveform)
                final_transcriptions.append(transcription)
                continue

            # Если аудио длинное, запускаем механизм чанкинга
            print(f"Processing long audio (duration: {audio_duration_s:.2f}s) by chunking...")
            segments = chunk_audio(
                length=audio_duration_s,
                segment_length=self.segment_length_s,
                segment_shift=self.segment_shift_s,
            )

            chunk_transcriptions = []
            for i, segment in enumerate(segments):
                print(f"  - Transcribing chunk {i+1}/{len(segments)} ({segment.start_time:.2f}s - {segment.end_time:.2f}s)")
                # Вырезаем нужный кусок аудио из исходного файла
                start_sample = int(segment.start_time * SAMPLING_RATE)
                end_sample = int(segment.end_time * SAMPLING_RATE)
                waveform_chunk = waveform[start_sample:end_sample]
                
                # Транскрибируем чанк
                chunk_text = self._transcribe_single_chunk(waveform_chunk)
                chunk_transcriptions.append(chunk_text)
            
            # Сшиваем транскрипции от всех чанков в одну
            print("  - Stitching transcriptions...")
            stitched_text = self._stitch_transcripts(chunk_transcriptions)
            final_transcriptions.append(stitched_text)

        return final_transcriptions