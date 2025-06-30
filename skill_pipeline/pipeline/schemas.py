"""
Data Schemas for the Job Analyzer Pipeline

This module defines the Pydantic models that serve as the data contracts for various
data structures used throughout the job analysis pipeline. Crucially, it documents
the schema for complex data stored in the database, ensuring that producers and
consumers of this data have a single source of truth.

The most important schema is `BaselineVectorSet`, which defines the structure of
the JSON data stored in the database, ensuring that producers and
consumers of this data have a single source of truth.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class VectorMetadata(BaseModel):
    total_vectors: int
    vector_dimension: int
    model_used: str

class RoleVectorData(BaseModel):
    role_vector: Optional[List[float]] = Field(None)
    category_vectors: Optional[Dict[str, List[float]]] = Field(None)
    skill_vectors: Dict[str, List[float]]
    vector_metadata: VectorMetadata
    
    # This is the unified output field we will use for analysis
    vectors: List[List[float]] = Field(description="A consolidated list of all skill vectors for a baseline")

class IndustryVectorData(BaseModel):
    industry_vector: Optional[List[float]] = Field(None)
    category_vectors: Optional[Dict[str, List[float]]] = Field(None)
    skill_vectors: Dict[str, List[float]]
    vector_metadata: VectorMetadata
    vectors: List[List[float]] = Field(description="A consolidated list of all skill vectors for a baseline")

class GlobalVectorData(BaseModel):
    global_vector: Optional[List[float]] = Field(None)
    skill_vectors: Dict[str, List[float]]
    vector_metadata: VectorMetadata
    vectors: List[List[float]] = Field(description="A consolidated list of all skill vectors for a baseline")

class JobContent(BaseModel):
    """职位内容的结构"""
    raw_text: str
    cleaned_text: str
    language: str

class BaselineVectorSet(BaseModel):
    """基线向量集，用于存储在数据库的JSON字段中"""
    vectors: List[List[float]]  # 向量列表, 现在可能只包含一个加权平均向量
    skill_map: Dict[str, int]   # 技能到其在原始（未加权）向量列表中索引的映射
    model: str                  # 使用的模型名称
    dimension: int              # 向量维度
    weights: Optional[Dict[str, float]] = None # 新增：存储每个技能的权重

class SkillAnalysisResult(BaseModel):
    """技能分析结果"""
    skill: str
    is_baseline: bool
    is_required: Optional[bool] = None
    similarity_score: Optional[float] = None
    is_synonym: bool = False
    is_common_skill: Optional[bool] = None

class AnomalyDetectionResult(BaseModel):
    """异常检测结果"""
    job_id: str
    anomaly_score: float
    missing_skills: List[SkillAnalysisResult]
    unexpected_skills: List[SkillAnalysisResult]
    required_skills_in_job: List[SkillAnalysisResult]
    baseline_role: Optional[str] = None
    baseline_industry: Optional[str] = None 