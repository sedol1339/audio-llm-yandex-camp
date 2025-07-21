from typing import List, Dict, Any
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import matplotlib.pyplot as plt
import numpy as np
import IPython.display
from tqdm.auto import tqdm
from termcolor import colored
from src.asr_eval.models.base import ASREvalWrapper
from abc import ABC, abstractmethod
from datasets import load_dataset, load_from_disk, Dataset
from src.asr_eval.segments.chunking import chunk_audio, average_segment_features
from src.asr_eval.segments.segment import AudioSegment
from hydra.core.hydra_config import HydraConfig
import pandas as pd
import os
from dataclasses import dataclass

@dataclass
class ExperimentResult:
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    model_params: Dict[str, Any]
    dataset_params: Dict[str, Any]
    processing_params: Dict[str, Any]
    timestamp: str

def create_result_row(
    model_name: str,
    dataset_name: str,
    model_cfg: DictConfig,
    dataset_cfg: DictConfig,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Создает структурированную строку для таблицы результатов"""
    return {
        "model": model_name,
        "dataset": dataset_name,
        **metrics,
        **{f"model_{k}": v for k, v in model_cfg.items() if k != "_target_"},
        **{f"dataset_{k}": v for k, v in dataset_cfg.items() if k not in ["path", "_target_"]},
    }


def save_results_to_table(results: List[ExperimentResult], output_dir: str):
    rows = []
    
    for result in results:
        row = {
            "timestamp": result.timestamp,
            "model": result.model_name,
            "dataset": result.dataset_name,
            **result.metrics,
            **{f"model_param_{k}": v for k, v in result.model_params.items()},
            **{f"dataset_param_{k}": v for k, v in result.dataset_params.items()},
            **{f"processing_param_{k}": v for k, v in result.processing_params.items()},
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Сохраняем в CSV
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

class AbsContext(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        ...

class StringContext(AbsContext):
    def __init__(self, string: str = ""):
        super().__init__()
        self.string = string
    def __call__(self, *args, **kwargs) -> str:
        return self.string

def get_prediction(model : ASREvalWrapper, dataset : Dataset, cfg : DictConfig, model_cfg : DictConfig, context : AbsContext = None):
    for sample in dataset:
        audio = sample['audio']['array']
        segments : List[AudioSegment] = chunk_audio(len(audio), cfg.segments.length, cfg.segments.shift, cfg.last_chunk_mode)
        features = []
        for segment in segments:
            audio_slise = audio[segment.slice(model_cfg.sample_rate)]
            features.append(model([audio_slise])[0])
        prediction = average_segment_features(segments, features, cfg.sefments.feature_tick_size, cfg.segments.averaging_weights)
    return prediction

def load_dataset_from_config(dataset_cfg: DictConfig) -> Dataset:
    pass

def initialize_model(model_cfg: DictConfig) -> ASREvalWrapper:
    pass

def compute_metrics(prediction, transcribtion, cfg) -> list:
    pass

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.run.output_dir, exist_ok=True)
        
    # context = AbsContext()
    all_results = []
    
    for dataset_name, dataset_cfg in cfg.datasets.items():
        print(f"\n=== Processing dataset: {dataset_name} ===")
        dataset = load_dataset_from_config(dataset_cfg)
        for model_name, model_cfg in cfg.models.items():
            print(f"\n  === Evaluating model: {model_name} on {dataset_name} ===")
            model = initialize_model(model_cfg)
            try:
                prediction = get_prediction(
                    model=model,
                    dataset=dataset,
                    model_cfg=model_cfg,
                    cfg=cfg
                    # context=context,
                )

                transcribtion = None
                
                metrics = compute_metrics(prediction, transcribtion, cfg)

                result_row = create_result_row(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    model_cfg=model_cfg,
                    dataset_cfg=dataset_cfg,
                    metrics=metrics,
                )

                all_results.append(result_row)
                
                print(f"  Successfully evaluated {model_name} on {dataset_name}")
            except Exception as e:
                print(f"  Error evaluating {model_name} on {dataset_name}: {str(e)}")
    
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(cfg.run.output_dir, "results.csv")
    results_df.to_csv(results_path, index=False)


if __name__ == "__main__":
    main()
