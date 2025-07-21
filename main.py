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

class AContext(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...

def get_prediction(model : ASREvalWrapper, dataset : Dataset, cfg : DictConfig, model_cfg : DictConfig, context : AContext = None):
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
    output_dir = HydraConfig.get().runtime.output_dir
    # os.makedirs(output_dir, exist_ok=True)
        
    # context = AContext()
    all_results = []
    metrics = []
    
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
                    processing_cfg=cfg.processing,
                    model_cfg=model_cfg,
                    # context=context,
                )

                transcribtion = None
                
                metrics.append(compute_metrics(prediction, transcribtion))
                
                print(f"  Successfully evaluated {model_name} on {dataset_name}")
            except Exception as e:
                print(f"  Error evaluating {model_name} on {dataset_name}: {str(e)}")
    
    # Сохраняем все результаты в CSV
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(results_path, index=False)


if __name__ == "__main__":
    main()
