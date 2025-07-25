from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Iterator, override

import torch
import soundfile as sf
from gigaam.vad_utils import segment_audio as gigaam_segment_audio

from ..segments.segment import AudioSegment
from ..utils.types import FLOATS


@contextmanager
def audio_as_file(waveform: FLOATS) -> Iterator[Path]:
    '''
    Turns an audio with sampling rate 16_000 into file, TODO check if it works
    with audio_as_file(waveform) as temp_file_path:
        recognize_speech(path=temp_file_path)
    '''
    with tempfile.NamedTemporaryFile('wb') as f:
        sf.write(f, waveform, samplerate=16_000, format='wav') # type: ignore
        yield Path(f.name)
    f.unlink()


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
    def __init__(self, **kwargs: Any):
        from .voxtral_wrapper import VoxtralmWrapper # type: ignore
        self.wrapper = VoxtralmWrapper(**kwargs) # type: ignore
    
    def run_audio_llm_inference(self, prompt: str, waveform: FLOATS) -> str:
        result = self.wrapper.transcribe(waveform, prompt=prompt) # type: ignore
        return result[0].text if result else "" # type: ignore
    

class Longform(ASREvalWrapper):
    def __init__(self, shortform_model: ASREvalWrapper, use_simple_segmentation: bool = False):
        self.model = shortform_model
        self.use_simple_segmentation = use_simple_segmentation
        
    def get_segment_boundaries(self, waveform: FLOATS) -> list[AudioSegment]:
        if self.use_simple_segmentation:
            print("Using simple segmentation")
            sample_rate = 16_000
            segment_duration = 30.0 
            segment_samples = int(segment_duration * sample_rate)
            
            segments = []
            total_samples = len(waveform)
            
            for start_sample in range(0, total_samples, segment_samples):
                end_sample = min(start_sample + segment_samples, total_samples)
                start_time = start_sample / sample_rate
                end_time = end_sample / sample_rate
                segments.append(AudioSegment(start_time, end_time)) # type: ignore
                
            return segments # type: ignore
        else:
            print("Using VAD-based segmentation")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            _, boundaries = gigaam_segment_audio(
                torch.tensor(waveform * 32768, dtype=torch.int16).clone(),
                16_000,
                max_duration=22.,
                min_duration=15.,
                new_chunk_threshold=0.2,
                device=device,
            )
            return [
                AudioSegment(segment_start, segment_end)
                for segment_start, segment_end in boundaries
            ]
    @override
    def transcribe(self, waveform: FLOATS, **kwargs: Any) -> list[TimedText]:
        segments = self.get_segment_boundaries(waveform)
        
        transcriptions: list[TimedText] = []
        for i, segment in enumerate(segments):
            segment_waveform = waveform[segment.slice()]
            
            if len(segment_waveform) < 100:
                print(f"Warning: Skipping segment {i+1} with {len(segment_waveform)} samples (too short)")
                continue
                
            preds = self.model.transcribe(segment_waveform, **kwargs)
            transcriptions += [p.shift(segment.start_time) for p in preds]
            
        return transcriptions


class RecurrentContextLongform(Longform):
    '''
    A wrapper around a shortform model that accepts 'prev_transcription' argument in `transcribe`
    '''
    def __init__(self, shortform_model: ASREvalWrapper, max_history_words: int = 100, use_simple_segmentation: bool = False):
        super().__init__(shortform_model, use_simple_segmentation)
        self.max_history_words = max_history_words
        self.use_simple_segmentation = use_simple_segmentation
    
    @override
    def transcribe(self, waveform: FLOATS, **kwargs: Any) -> list[TimedText]:
        segments = self.get_segment_boundaries(waveform)
        
        transcriptions: list[TimedText] = []
        for i, segment in enumerate(segments):
            segment_waveform = waveform[segment.slice()]
            
            if len(segment_waveform) < 100:
                print(f"Warning: Skipping segment {i+1} with {len(segment_waveform)} samples (too short)")
                continue
            
            full_history = ' '.join(t.text for t in transcriptions)
            words = full_history.split()
            if len(words) > self.max_history_words:
                limited_history = ' '.join(words[-self.max_history_words:])
            else:
                limited_history = full_history
                
            preds = self.model.transcribe(
                segment_waveform,
                prev_transcription=limited_history,
                **kwargs
            )
            transcriptions += [p.shift(segment.start_time) for p in preds]
            
        return transcriptions

'''
Example usage:
shortform_llm = Voxtral()
longform_recognizer = RecurrentContextLongform(shortform_llm)
longform_recognizer.transcribe(long_waveform, domain_words='<chemistry text>')
'''