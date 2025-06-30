from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ModelThresholds:
    min_avg_similarity: float = 0.35
    similarity_threshold: float = 0.1
    idf_threshold: float = 2.5
    role_similarity_threshold: float = 0.3
    role_title_weight: float = 0.9
    industry_similarity_threshold: float = 0.3

@dataclass
class CacheConfig:
    cache_size: int = 1000

@dataclass
class PipelineConfig:
    batch_size: int = 32

@dataclass
class AppConfig:
    model_thresholds: ModelThresholds = field(default_factory=ModelThresholds)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    model_name: str = 'all-MiniLM-L6-v2'
    spacy_model_name: str = 'en_core_web_sm'

def get_config() -> AppConfig:
    # In the future, this can be extended to load from a YAML file or environment variables.
    return AppConfig() 