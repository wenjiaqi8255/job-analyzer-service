#!/usr/bin/env python3
import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """管线配置类"""
    # 模型配置
    classification_model: str = 'MohammedDhiyaEddine/job-skill-sentence-transformer-tsdae'
    backup_model: str = 'all-MiniLM-L6-v2'
    
    # 分级阈值配置 - 更严格的阈值设置
    high_confidence_threshold: float = 0.75  # 高置信度阈值
    medium_confidence_threshold: float = 0.55  # 中等置信度阈值  
    low_confidence_threshold: float = 0.35  # 低置信度阈值
    
    # 兼容旧接口的阈值
    role_threshold: float = 0.55  # 提高默认角色阈值
    industry_threshold: float = 0.45  # 提高默认行业阈值
    
    # 基线生成配置
    min_skills_for_baseline: int = 5
    embedding_dimension: int = 768
    
    # 数据库配置
    supabase_url: str = None
    supabase_key: str = None
    
    # 输出配置
    output_dir: str = "output"
    backup_existing: bool = True
    
    # 调试配置
    enable_debug_logging: bool = False
    save_classification_details: bool = True
    
    def __post_init__(self):
        """加载环境变量和验证配置"""
        load_dotenv()
        if not self.supabase_url:
            self.supabase_url = os.getenv("SUPABASE_URL")
        if not self.supabase_key:
            self.supabase_key = os.getenv("SUPABASE_KEY")
            
        # 验证阈值合理性
        if self.high_confidence_threshold <= self.medium_confidence_threshold:
            logger.warning("⚠️ 高置信度阈值应该大于中等置信度阈值")
        if self.medium_confidence_threshold <= self.low_confidence_threshold:
            logger.warning("⚠️ 中等置信度阈值应该大于低置信度阈值") 