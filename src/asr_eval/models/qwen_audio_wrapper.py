import torch
from typing import override

# Критически важные импорты
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# Ваши базовые классы
from .base import ASREvalWrapper
from ..utils.types import FLOATS

class QwenAudioWrapper(ASREvalWrapper):
    """
    Улучшенная обертка для Qwen/Qwen2-Audio-7B-Instruct.
    
    ИЗМЕНЕНИЯ:
    - Метод __call__ переписан для обработки аудиофайлов в цикле, по аналогии
      с WhisperLongformWrapper, для повышения качества.
    - В model.generate() добавлены параметры для детерминированной генерации,
      что критически важно для задачи транскрибации.
    - Улучшен текстовый промпт для более точного следования инструкции.
    """
    def __init__(
        self,
        model_name: str = 'Qwen/Qwen2-Audio-7B-Instruct',
        # ИЗМЕНЕНИЕ: Более точный промпт для лучшего качества
        transcription_prompt: str = "Дословно и высокоточно транскрибируй на русский язык следующую аудиозапись на русском языке, не добовляя комментари, не исправляя ошибки говорящего и не теряя слова. Твой ответ должен содержать только итоговый текст транскрибации на русском языке без потери слов и без добавления новых слов."
    ):
        self.model_name = model_name
        self.transcription_prompt = transcription_prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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

    @override
    @torch.inference_mode()
    def __call__(self, waveforms: list[FLOATS]) -> list[str]:
        """
        Принимает список аудиофрагментов и возвращает их транскрипции.
        
        ИЗМЕНЕНИЕ: Обработка происходит в цикле для каждого аудиофайла,
        чтобы применить индивидуальные и точные настройки генерации.
        """
        self._maybe_instantiate()
        if self.model is None or self.processor is None:
            raise RuntimeError("Model or processor not instantiated.")

        transcriptions = []
        # ИЗМЕНЕНИЕ: Обрабатываем каждый waveform в цикле, как в Whisper
        for i, waveform in enumerate(waveforms):
            print(f"Processing waveform {i+1}/{len(waveforms)}...")

            # 1. Готовим диалог для одного аудиофайла
            conversation = [
                {"role": "user", "content": [
                    {"type": "text", "text": self.transcription_prompt},
                    {"type": "audio"},
                ]}
            ]
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )

            # 2. Процессор принимает текст и ОДНО аудио
            inputs = self.processor(
                text=[prompt], # передаем как список из одного элемента
                audios=[waveform], # передаем как список из одного элемента
                return_tensors="pt",
                padding=True # здесь паддинг не играет роли, так как в пакете 1 элемент
            ).to(self.device) # type: ignore

            # 3. Генерируем текст с параметрами для высокого качества
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                # ИЗМЕНЕНИЕ: Добавляем ключевые параметры для качества
                do_sample=False, # Отключаем сэмплирование, выбираем самые вероятные токены
                temperature=0.0, # Температура 0 для полной детерминированности
            )

            # 4. Декодируем результат
            input_token_len = inputs.input_ids.shape[1]
            output_ids = generate_ids[:, input_token_len:]
            
            result_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0] # Берем первый (и единственный) элемент из пакета
            
            transcriptions.append(result_text)

        return transcriptions
