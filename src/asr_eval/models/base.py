from dataclasses import dataclass
from typing import Any, override

import torch
from gigaam.vad_utils import segment_audio as gigaam_segment_audio

from ..segments.segment import AudioSegment
from ..utils.types import FLOATS


@dataclass(frozen=True)
class TimedText(AudioSegment):
    text: str


class ASREvalWrapper:
    '''
    A wrapper that processes audios with sampling rate 16_000, of float dtype.
    '''
    def transcribe(self, waveform: FLOATS, **kwargs: Any) -> list[TimedText]:
        raise NotImplementedError()


class AbstractShortformAudioLLM(ASREvalWrapper):
    @override
    def transcribe(
        self,
        waveform: FLOATS,
        prev_transcription: str | None = None,
        domain_words: str | None = None,
        **kwargs: Any,
    ) -> list[TimedText]:
        prompt = 'Recognize the audio {{audio}}.'
        if prev_transcription:
            prompt += f' The previous transcription was: {prev_transcription}.'
        if domain_words:
            prompt += f' The following text may be related: {domain_words}.'
        text = self.run_audio_llm_inference(prompt, waveform)
        return [TimedText(0, len(waveform) / 16_000, text)]
    
    def run_audio_llm_inference(self, prompt: str, waveform: FLOATS) -> str:
        raise NotImplementedError()


class QwenAudio(AbstractShortformAudioLLM):
    def run_audio_llm_inference(self, prompt: str, waveform: FLOATS) -> str:
        ... # TODO


class Voxtral(AbstractShortformAudioLLM):
    def run_audio_llm_inference(self, prompt: str, waveform: FLOATS) -> str:
        ... # TODO


class Longform(ASREvalWrapper):
    def __init__(self, shortform_model: ASREvalWrapper):
        self.model = shortform_model
        
    def get_segment_boundaries(self, waveform: FLOATS) -> list[AudioSegment]:
        _, boundaries = gigaam_segment_audio(
            torch.tensor(waveform * 32768, dtype=torch.int16).clone(),
            16_000,
            max_duration=22.,
            min_duration=15.,
            new_chunk_threshold=0.2,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        return [
            AudioSegment(segment_start, segment_end)
            for segment_start, segment_end in boundaries
        ]
    
    @override
    def transcribe(self, waveform: FLOATS, **kwargs: Any) -> list[TimedText]:
        segments = self.get_segment_boundaries(waveform)
        
        transcriptions: list[TimedText] = []
        for segment in segments:
            preds = self.model.transcribe(waveform[segment.slice()], **kwargs)
            transcriptions += [p.shift(segment.start_time) for p in preds]
            
        return transcriptions


class RecurrentContextLongform(Longform):
    '''
    A wrapper around a shortform model that accepts 'prev_transcription' argument in `transcribe`
    '''
    @override
    def transcribe(self, waveform: FLOATS, **kwargs: Any) -> list[TimedText]:
        segments = self.get_segment_boundaries(waveform)
        
        transcriptions: list[TimedText] = []
        for segment in segments:
            preds = self.model.transcribe(
                waveform[segment.slice()],
                prev_transcription=' '.join(t.text for t in transcriptions),
                **kwargs
            )
            transcriptions += [p.shift(segment.start_time) for p in preds]
            
        return transcriptions

'''
Example usage:
shortform_llm = QwenAudio()
longform_recognizer = RecurrentContextLongform(shortform_llm)
longform_recognizer.transcribe(long_waveform, domain_words='<chemistry text>')
'''