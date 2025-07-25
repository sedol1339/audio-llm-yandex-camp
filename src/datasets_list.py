from typing import cast
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets, Audio # type: ignore


# A set of ASR test datasets

def load_youtube_lectures() -> Dataset:
    # "train" is a single split here
    # loading dangrebenkin/long_audio_youtube_lectures from HF gives an error with datasets==3.6.0
    # https://github.com/huggingface/datasets/issues/7676
    return cast(
        Dataset,
        load_dataset('dangrebenkin/long_audio_youtube_lectures', split='train')
        .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
    )
    # return cast(Dataset, load_from_disk('/asr_datasets/long_audio_youtube_lectures'))

def load_golos_farfield() -> Dataset:
    return cast(
        Dataset,
        load_dataset('bond005/sberdevices_golos_100h_farfield', split='test')
        .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
    )

def load_rulibrispeech() -> Dataset:
    return cast(
        Dataset,
        load_dataset('bond005/rulibrispeech', split='test')
        .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
    )

def load_podlodka() -> Dataset:
    return cast(
        Dataset,
        load_dataset('bond005/podlodka_speech', split='test')
        .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
    )

def load_podlodka_full() -> Dataset:
    return concatenate_datasets([
        cast(
            Dataset,
            load_dataset('bond005/podlodka_speech', split='test')
            .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
        ),
        cast(
            Dataset,
            load_dataset('bond005/podlodka_speech', split='train')
            .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
        ),
        cast(
            Dataset,
            load_dataset('bond005/podlodka_speech', split='validation')
            .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
        ),
    ])

def load_sova_rudevices() -> Dataset:
    return cast(
        Dataset,
        load_dataset('bond005/sova_rudevices', split='test')
        .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
    )

def load_resd() -> Dataset:
    return (
        cast(Dataset, load_dataset('Aniemore/resd_annotated', split='test'))
        .rename_column('text', 'transcription')
        .rename_column('speech', 'audio')
        .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
    )

def load_fleurs() -> Dataset:
    return (
        cast(Dataset, load_dataset(
            'google/fleurs',
            name='ru_ru',
            split='test',
            trust_remote_code=True,
        ))
        .remove_columns('transcription')
        .rename_column('raw_transcription', 'transcription')
        .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
    )

def load_speech_massive() -> Dataset:
    return (
        cast(Dataset, load_dataset(
            'FBK-MT/Speech-MASSIVE-test',
            name='ru-RU',
            split='test',
        ))
        .rename_column('utt', 'transcription')
        .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
    )

def load_common_voice_17_0() -> Dataset:
    return (
        cast(Dataset, load_dataset(
            'mozilla-foundation/common_voice_17_0',
            name='ru',
            split='test',
            trust_remote_code=True,
        ))
        .rename_column('sentence', 'transcription')
        .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
    )

def load_wikipedia_asr_splitted() -> Dataset:
    # Загружаем датасет с отключенной проверкой аудио
    ds_dict = load_dataset(
        'rmndrnts/wikipedia_asr_splitted',
        verification_mode="no_checks"
    )
    
    # Объединяем все сплиты в один датасет (исправленная строка)
    combined = concatenate_datasets([ds_dict[split] for split in ds_dict.keys()])
    
    # Фильтруем нежелательные категории
    categories_to_remove = {'politics', 'economics', 'philosophy'}
    filtered_indices = [
        i for i, x in enumerate(combined['category']) 
        if x not in categories_to_remove
    ]
    filtered = combined.select(filtered_indices)
    
    # Преобразуем аудио
    filtered = filtered.cast_column('audio', Audio(sampling_rate=16000))
    
    return filtered

def load_rmndrnts_lena_dataset() -> Dataset:
    # Загружаем DatasetDict
    dataset_dict = cast(DatasetDict, load_dataset('rmndrnts/lena_dataset_splitted'))
    
    # Собираем все поднаборы в один список
    all_datasets = [dataset_dict[subset] for subset in dataset_dict.keys()]
    
    # Объединяем все датасеты в один
    combined_dataset = concatenate_datasets(all_datasets)
    
    # Применяем преобразование аудио
    combined_dataset = combined_dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    return combined_dataset