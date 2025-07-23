from typing import cast
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets 
from datasets import Dataset, Features, Value, Sequence, Audio, load_from_disk
from datasets import load_dataset, Dataset, Audio, Features, Value
from typing import cast
import numpy as np
import librosa
from datasets import Dataset, Audio, Features, Value, load_dataset
from typing import cast


# A set of ASR test datasets

def load_youtube_lectures() -> Dataset:
    SAMPLING_RATE = 16000
    ds = cast(Dataset, load_dataset(
        'dangrebenkin/long_audio_youtube_lectures',
        split='train'
    ))
    samples = []

    for i, sample in enumerate(ds):
        s = {}
        s['transcription'] = sample['transcription']
        s['context'] = sample['domain']
        
        audio_data = sample['audio']
        
        if isinstance(audio_data['array'], list):
            waveform = np.array(audio_data['array'], dtype=np.float32)
            sr = audio_data['sampling_rate']
            
            # Ресемплируем если нужно
            if sr != SAMPLING_RATE:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLING_RATE)
                sr = SAMPLING_RATE
        else:
            waveform = audio_data['array']
            sr = audio_data['sampling_rate']
        
        s['audio'] = {
            'array': waveform,
            'sampling_rate': sr
        }
        
        samples.append(s)


    dataset = Dataset.from_dict({
        'audio': [s['audio'] for s in samples],
        'transcription': [s['transcription'] for s in samples],
        'context': [s['context'] for s in samples]
    }, features=Features({
        'audio': Audio(sampling_rate=SAMPLING_RATE),
        'transcription': Value('string'),
        'context': Value('string')
    }))
    
    return dataset


def load_golos_farfield() -> Dataset:
    SAMPLING_RATE = 16000
    ds = cast(Dataset, load_dataset(
        'bond005/sberdevices_golos_100h_farfield',
        split='test'
    ))
    samples = []

    for i, sample in enumerate(ds):
        s = {}
        s['transcription'] = sample['transription']

        if sample['audio']['sampling_rate'] != 16000:
            audio_path = sample['audio']
            waveform, _ = librosa.load(audio_path, sr=SAMPLING_RATE)
            s['audio'] =  waveform

        samples.append(s)

    dataset = Dataset.from_list(samples, features=Features({ # type: ignore
        'audio': Audio(decode=True),
        'transcription': Value('string')
    }))
    return dataset



def load_resd() -> Dataset:
    SAMPLING_RATE = 16000
    ds = (
        cast(Dataset, load_dataset('Aniemore/resd_annotated', split='test'))
        .rename_column('text', 'transcription')
        .rename_column('speech', 'audio')
    )
    samples = []

    for sample in ds:
        s = {}
        s['transcription'] = sample['transcription']
        s['context'] = ''
        
        # Получаем аудиоданные
        audio_data = sample['audio']
        
        if audio_data['sampling_rate'] != SAMPLING_RATE:
            # Если частота не 16 кГц, ресемплируем массив напрямую
            waveform = librosa.resample(
                audio_data['array'], 
                orig_sr=audio_data['sampling_rate'], 
                target_sr=SAMPLING_RATE
            )
            s['audio'] = {
                'array': waveform,
                'sampling_rate': SAMPLING_RATE,
                'path': audio_data.get('path', '')  # Сохраняем путь, если есть
            }
        else:
            # Если частота уже 16 кГц, используем как есть
            s['audio'] = audio_data
        

            
        samples.append(s)

    dataset = Dataset.from_list(samples, features=Features({
        'audio': Audio(sampling_rate=SAMPLING_RATE),
        'transcription': Value('string'),
        'context': Value('string')
    }))
    return dataset
