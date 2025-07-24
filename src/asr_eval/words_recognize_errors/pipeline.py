from pprint import pprint

import jiwer
from numpy.random import choice

from src.asr_eval.align.parsing import split_text_into_tokens
from src.asr_eval.align.recursive import align

from ..models.flamingo_wrapper import FlamingoWrapper
from .domain_words import get_domain_words
from .generate_audio import generate_audio
from .generate_texts import generate_texts


class ErrorDictPipeline:
    def __init__(self):
        self.stt = FlamingoWrapper()

    def _get_domain_words(self, topic: str, random_size: int | None = 16) -> list[str]:
        words: list[str] | None = choice(get_domain_words(), size=random_size).tolist()
        if words is None:
            return list()
        return words

    def _generate_texts(self, words: list[str]) -> list[str]:
        texts: list[str] = generate_texts(words)
        return texts

    def _generate_audio(self, texts: list[str]) -> list[str]:
        audios: list[str] = generate_audio(texts)
        return audios

    def _recognize_audio(self, audios: list[str]) -> list[str]:
        trans = self.stt.transcribe_from_file(
            audios,
            prompt="Транскрибируй это аудио полностью на русском. Приведи полный текст который приведен произносится в этом аудио на русском языке",
        )
        return [t.text for t in trans]

    def _get_wer_for_texts(
        self, true_text: list[str], recognized_text: list[str]
    ) -> list[float]:
        min_len = min(len(true_text), len(recognized_text))
        return [jiwer.wer(true_text[i], recognized_text[i]) for i in range(min_len)]

    def _get_wrong_recognized_words(
        self, true_text: list[str], recognized_text: list[str]
    ) -> dict[str, str]:
        min_len = min(len(true_text), len(recognized_text))
        result = []
        for i in range(min_len):
            true_tokens = split_text_into_tokens(true_text[i])
            pred_tokens = split_text_into_tokens(recognized_text[i])
            result.append(align(true_tokens, pred_tokens))
        for i, mlist in enumerate(result):
            result[i] = [
                m
                for m in mlist.matches
                if m.true is not None
                and m.pred is not None
                and m.true.value != m.pred.value
            ]
        return result

    def _get_fix_dict(self, errors) -> list[dict[str, str]]:
        return [{w.pred.value: w.true.value for w in e} for e in errors]

    def run(self, topik: str | None = None) -> dict[str, str]:
        words: list[str] = self._get_domain_words(topik)
        texts: list[str] = self._generate_texts(words)
        audios: list[str] = self._generate_audio(texts)
        recognized_text: list[str] = self._recognize_audio(audios)
        wers: list[float] = self._get_wer_for_texts(texts, recognized_text)
        errors = self._get_wrong_recognized_words(texts, recognized_text)
        fix_dict = self._get_fix_dict(errors)


        for i in range(len(texts)):
            print(f"{words[i]}:")
            print("\tSource text:  " + texts[i])
            print("\tRecognized text:  " + recognized_text[i])
            print("\tFix dict:  ", fix_dict[i])
            print(f"\tWer:{wers[i]}\n\n")


if __name__ == "__main__":
    pipe = ErrorDictPipeline()
    pipe.run()
