from typing import override
import torch
from transformers import AutoProcessor
from models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration

from .base import ASREvalWrapper
from ..utils.types import FLOATS


class Qwen2AudioWrapper(ASREvalWrapper):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        max_new_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _maybe_instantiate(self):
        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.model.eval()

    @override
    def __call__(self, waveforms: list[FLOATS]) -> list[str]:
        self._maybe_instantiate()
        results: list[str] = []

        for waveform in waveforms:
            if isinstance(waveform, list):
                waveform = torch.tensor(waveform, dtype=torch.float32)
            elif isinstance(waveform, torch.Tensor):
                waveform = waveform.float()
            else:
                raise TypeError("Unsupported waveform type")

            waveform = waveform.squeeze()
            sr = 16000

            inputs = self.processor(
                audios=waveform,
                sampling_rate=sr,
                return_tensors="pt"
            ).to(self.device)

            prompt = "<|im_start|>user\nWhat did the audio say?<|im_end|>\n<|im_start|>assistant\n"
            input_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    audio_values=inputs["input_values"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                )

            text = self.processor.batch_decode(outputs[:, input_ids.shape[-1]:], skip_special_tokens=True)[0]
            results.append(text.strip())

        return results
