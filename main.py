from typing import List, Dict, Any, Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from src.asr_eval.models.base import ASREvalWrapper
from abc import ABC, abstractmethod
from datasets import load_dataset, load_from_disk, Dataset
from src.asr_eval.segments.chunking import chunk_audio, average_segment_features
from src.asr_eval.segments.segment import AudioSegment
import pandas as pd
import os
from dataclasses import dataclass
from src.datasets_list import load_podlodka
from src.asr_eval.models.whisper_wrapper import WhisperLongformWrapper
from src.asr_eval.align.parsing import parse_multivariant_string, split_text_into_tokens
from src.asr_eval.align.recursive import align
import logging
from omegaconf import open_dict
import json
from src.asr_eval.utils.types import FLOATS
import torchaudio

# Настройка логгера
logger = logging.getLogger(__name__)

@dataclass
class ExperimentResult:
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    model_params: Dict[str, Any]
    dataset_params: Dict[str, Any]

def create_result_row(
    model_name: str,
    dataset_name: str,
    model_cfg: DictConfig,
    dataset_cfg: DictConfig,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "model": model_name,
        "dataset": dataset_name,
        **metrics,
        **{f"model_{k}": v for k, v in model_cfg.items() if k != "_target_"},
        **{f"dataset_{k}": v for k, v in dataset_cfg.items() if k not in ["path", "_target_"]},
    }

class AbsContext(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass

class StringContext(AbsContext):
    def __init__(self, string: str = ""):
        self.string = string
    
    def __call__(self, *args, **kwargs) -> str:
        return self.string

def get_predictions(
    model: ASREvalWrapper, 
    dataset: Dataset, 
    cfg: DictConfig, 
    model_cfg: DictConfig,
    context: Optional[AbsContext] = StringContext("")
) -> List[str]:
    predictions = []
    for sample in dataset:
        audio: FLOATS = sample['audio']['array']
        print(len(audio))
        sample_rate: int = sample['audio']['sampling_rate']
        transforms = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=cfg.sample_rate)
        audio = transforms(audio)    
        segments: List[AudioSegment] = chunk_audio(
            len(audio),
            cfg.segments.length,
            cfg.segments.shift,
            cfg.segments.last_chunk_mode
        )
        
        features = []
        for segment in segments:
            audio_slice = audio[segment.slice(sample_rate)]
            print(len(audio_slice), segment.duration)
            features.append(model([audio_slice])[0])
        
        prediction = average_segment_features(
            segments,
            features,
            cfg.segments.feature_tick_size,
            cfg.segments.averaging_weights
        )
        predictions.append(prediction)
    
    return predictions

def load_dataset_from_config(dataset_cfg: DictConfig) -> Dataset:
    """Загружает датасет по имени конфига"""
    if dataset_cfg.tag== "podlodka":
        return load_podlodka()
    # Добавьте здесь загрузку других датасетов
    raise ValueError(f"Unknown dataset: {dataset_cfg.tag}")

def initialize_model(model_cfg: DictConfig) -> ASREvalWrapper:
    if model_cfg.model == "whisper":
        return WhisperLongformWrapper(model_cfg.name)
    # Добавьте здесь инициализацию других моделей
    raise ValueError(f"Unknown model: {model_cfg.model}")

def compute_metrics(predictions: List[str], transcriptions: List[str], cfg: DictConfig) -> Dict[str, float]:
    metrics_dict = {}
    
    if "wer" in cfg.metrics_list:
        wer = 0.0
        count = 0
        for prediction, transcription in zip(predictions, transcriptions):
            true_words = parse_multivariant_string(transcription)
            alignment = align(true_words, split_text_into_tokens(prediction))
            wer += alignment.score.n_word_errors / max(1, len(true_words))
            count += 1
        metrics_dict["wer"] = wer / max(1, count)
    
    return metrics_dict

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print("=== Current Configuration ===")
    print(json.dumps(cfg_dict, indent=4, ensure_ascii=False))
    os.makedirs(cfg.run.output_dir, exist_ok=True)
    all_results = []
    
    # Итерация по всем датасетам в конфиге
    for dataset_name in cfg.datasets:
        dataset_cfg = cfg.datasets[dataset_name]
        logger.info(f"\n=== Processing dataset: {dataset_name} ===")
        
        try:
            dataset = load_dataset_from_config(dataset_cfg)
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            continue
        
        # Итерация по всем моделям
        for model_name in cfg.models:
            model_cfg = cfg.models[model_name]
            logger.info(f"\n  === Evaluating model: {model_name} on {dataset_name} ===")
            
            try:
                model = initialize_model(model_cfg)
                predictions = get_predictions(
                    model=model,
                    dataset=dataset,
                    cfg=cfg,
                    model_cfg=model_cfg
                )
                
                transcriptions = dataset['transcription']
                
                metrics = compute_metrics(predictions, transcriptions, cfg)
                
                result_row = create_result_row(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    model_cfg=model_cfg,
                    dataset_cfg=dataset_cfg,
                    metrics=metrics,
                )
                all_results.append(result_row)
                
                logger.info(f"  Successfully evaluated {model_name} on {dataset_name}")
            except Exception as e:
                logger.exception(f"Error evaluating {model_name} on {dataset_name}")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(cfg.run.output_dir, "results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
    else:
        logger.error("No results were generated")

if __name__ == "__main__":
    main()