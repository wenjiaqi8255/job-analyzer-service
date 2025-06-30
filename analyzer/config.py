from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ModelThresholds:
    min_avg_similarity: float = 0.35
    similarity_threshold: float = 0.1
    idf_threshold: float = 2.5
    role_similarity_threshold: float = 0.35
    role_title_weight: float = 0.9
    industry_similarity_threshold: float = 0.4
    
    # New thresholds for the refined anomaly detection logic
    core_skill_threshold: float = 0.4  # Similarity to be considered a core skill (and not an anomaly)
    boilerplate_threshold: float = 0.7 # Similarity to be considered standard boilerplate (and not an anomaly)
    cross_role_threshold: float = 0.4 # Similarity to another role to be flagged as a cross-role anomaly

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