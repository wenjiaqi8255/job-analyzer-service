"""
Data Schemas for the Job Analyzer Pipeline

This module defines the Pydantic models that serve as the data contracts for various
data structures used throughout the job analysis pipeline. Crucially, it documents
the schema for complex data stored in the database, ensuring that producers and
consumers of this data have a single source of truth.

The most important schema is `BaselineVectorSet`, which defines the structure of
the JSON data stored in the `vectors_data` column of the `baseline_vectors` table.
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

class BaselineVectorSet(BaseModel):
    """
    This model represents the structure of the `vectors_data` column 
    in the `baseline_vectors` table.

    This is the definitive schema for the JSON object stored in the database.
    Any process that reads from or writes to this column should adhere to this
    structure. The `ResourceManager` in the `analyzer` service, for example,
    parses this structure to load baseline vectors for anomaly detection.
    """
    vectors: List[List[float]] = Field(description="Consolidated list of vectors for analysis")
    skill_map: Dict[str, int] = Field(description="Mapping from skill string to its index in the vectors list")
    model: str
    dimension: int 