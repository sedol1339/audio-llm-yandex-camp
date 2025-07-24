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
from src.datasets_list import *
from src.asr_eval.models.whisper_wrapper import WhisperLongformWrapper
from src.asr_eval.align.parsing import parse_multivariant_string, split_text_into_tokens
from src.asr_eval.align.recursive import align
import logging
from omegaconf import open_dict
import json
from src.asr_eval.utils.types import FLOATS
import torchaudio
from src.serialize import save_to_json
from tqdm import tqdm
from collections import OrderedDict
from src.asr_eval.models.qwen_audio_wrapper import QwenAudioWrapper
from src.asr_eval.models.voxtral_wrapper import VoxtralmWrapper
from src.asr_eval.models.gigaam_wrapper import GigaAMWrapper
from src.asr_eval.models.gemma_wrapper import Gemma3nSTTWrapper
# from src.asr_eval.models.t_one_wrapper import TOneWrapper

# Настройка логгера
logger = logging.getLogger(__name__)

def save_predict(index, model_cfg, dataset_name, prediction, transcription, cfg: DictConfig):
    save_dict = {}
    true_words = parse_multivariant_string(transcription)
    alignment = align(true_words, split_text_into_tokens(prediction))
    save_dict['preiction'] = prediction
    save_dict['transcription'] = transcription
    save_dict['alignment'] = alignment
    save_to_json(save_dict, os.path.join(cfg.output_dir, cfg.run.name, f"{dataset_name}_{model_cfg.model_name}", f"{index}.json"))    
        

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
        **{f"model_{k}": v for k, v in model_cfg.params.items() if k != "_target_"},
        **{f"dataset_{k}": v for k, v in dataset_cfg.params.items() if k not in ["path", "_target_"]},
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
    dataset_cfg: DictConfig,
    context: Optional[AbsContext] = StringContext("")
) -> List[str]:
    predictions = []
    for index, sample in tqdm(enumerate(dataset)):
        audio: FLOATS = sample['audio']['array']
        sample_rate: int = sample['audio']['sampling_rate']
        transforms = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=cfg.sample_rate)
        audio = transforms(audio)
        prediction = model([audio])[0]
        predictions.append(prediction)
        transcription = sample['transcription']
        save_predict(index, model_cfg, dataset_cfg.dataset, prediction, transcription, cfg)
    return predictions

def load_dataset_from_config(dataset_cfg: DictConfig) -> Dataset:
    """Загружает датасет по имени конфига"""
    if dataset_cfg.dataset == "podlodka":
        return load_podlodka()
    if dataset_cfg.dataset == "youtube_lectures":
        return load_youtube_lectures()
    if dataset_cfg.dataset == "golos_farfield":
        return load_golos_farfield()
    if dataset_cfg.dataset == "rulibrispeech":
        return load_rulibrispeech()
    if dataset_cfg.dataset == "sova_rudevices":
        return load_sova_rudevices()
    if dataset_cfg.dataset == "fleurs":
        return load_fleurs()
    if dataset_cfg.dataset == "resd":
        return load_resd()
    if dataset_cfg.dataset == "common_voice_17_0":
        return load_common_voice_17_0()
    # Добавьте здесь загрузку других датасетов
    raise ValueError(f"Unknown dataset: {dataset_cfg.dataset}")

def initialize_model(model_cfg: DictConfig) -> ASREvalWrapper:
    if model_cfg.model == "whisper":
        return WhisperLongformWrapper(**model_cfg.params)
    if model_cfg.model == "qwen_audio":
        return QwenAudioWrapper(**model_cfg.params)
    if model_cfg.model == "woxtral":
        return VoxtralmWrapper(**model_cfg.params)
    if model_cfg.model == "gigaam":
        return GigaAMWrapper(**model_cfg.params)
    if model_cfg.model == "gemma":
        return Gemma3nSTTWrapper(**model_cfg.params)
    if model_cfg.model == "t_one":
        return TOneWrapper(**model_cfg.params)
        
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
    os.makedirs(cfg.output_dir, exist_ok=True)
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
            logger.info(f"\n=== Evaluating model: {model_cfg.model_name} on {dataset_name} params ===")
            
            try:
                model = initialize_model(model_cfg)
                predictions = get_predictions(
                    model=model,
                    dataset=dataset,
                    cfg=cfg,
                    model_cfg=model_cfg,
                    dataset_cfg=dataset_cfg
                )
                
                transcriptions = dataset['transcription']
                
                logger.info(f"\n=== Compute metrics: {model_name} on {dataset_name} ===")
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
        results_path = os.path.join(cfg.output_dir, cfg.run.name, "results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
    else:
        logger.error("No results were generated")

if __name__ == "__main__":
    main()
