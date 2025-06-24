import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import PipelineConfig
from .schemas import BaselineVectorSet

logger = logging.getLogger(__name__)


class BaselineGenerator:
    """语义基线构建器"""

    def __init__(self, config: PipelineConfig, classification_model: SentenceTransformer):
        self.config = config
        self.classification_model = classification_model  # 用于分类的模型

        # 初始化专门用于向量生成的模型
        logger.info("🔧 初始化向量生成模型...")
        try:
            self.vector_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ 成功加载向量生成模型: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"❌ 无法加载向量生成模型: {e}")
            raise

    def generate_all_baselines(self, role_classifications: Dict, industry_classifications: Dict) -> Dict:
        """生成所有基线（分离语义数据和向量数据）"""
        logger.info("🏗️ 开始生成语义基线...")

        baselines = {
            'global': self._generate_global_baseline(role_classifications, industry_classifications),
            'roles': self._generate_role_baselines(role_classifications),
            'industries': self._generate_industry_baselines(industry_classifications)
        }

        logger.info("✅ 所有基线生成完成")
        return baselines

    def _generate_global_baseline(self, role_classifications: Dict, industry_classifications: Dict) -> Dict:
        """生成全局基线"""
        logger.info("🌍 生成全局基线...")

        all_skills = set()
        for categories in role_classifications.values():
            for skills in categories.values():
                all_skills.update(skills)

        for categories in industry_classifications.values():
            for skills in categories.values():
                all_skills.update(skills)

        all_skills = list(all_skills)

        # 统计信息
        role_distribution = {role: sum(len(skills) for skills in categories.values())
                             for role, categories in role_classifications.items()}
        industry_distribution = {industry: sum(len(skills) for skills in categories.values())
                                 for industry, categories in industry_classifications.items()}

        # 分离语义数据和向量数据
        semantic_data = {
            'total_skills': len(all_skills),
            'skills_list': sorted(all_skills),  # 技能列表（人类可读）
            'statistics': {
                'role_distribution': role_distribution,
                'industry_distribution': industry_distribution,
                'total_roles': len(role_classifications),
                'total_industries': len(industry_classifications)
            },
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'classification_model': self.config.classification_model,
                'vector_model': 'all-MiniLM-L6-v2',
                'embedding_dimension': 384  # all-MiniLM-L6-v2 的维度
            }
        }

        # 生成向量数据
        vector_data = self._generate_vectors_for_skills(all_skills)

        return {
            'semantic_data': semantic_data,
            'vector_data': vector_data
        }

    def _generate_role_baselines(self, role_classifications: Dict) -> Dict:
        """生成角色基线"""
        logger.info("👔 生成角色基线...")

        role_baselines = {}

        for role, categories in role_classifications.items():
            all_role_skills = []
            for skills in categories.values():
                all_role_skills.extend(skills)

            if len(all_role_skills) < self.config.min_skills_for_baseline:
                logger.warning(f"角色 {role} 技能数量不足，跳过基线生成")
                continue

            # 语义数据（人类可读）
            semantic_data = {
                'role_name': role,
                'categories': categories,
                'total_skills': len(all_role_skills),
                'skills_list': sorted(list(set(all_role_skills))),
                'statistics': {
                    'category_distribution': {cat: len(skills) for cat, skills in categories.items() if skills}
                },
                'metadata': {
                    'generation_date': datetime.now().isoformat(),
                    'classification_model': self.config.classification_model,
                    'vector_model': 'all-MiniLM-L6-v2'
                }
            }

            # 向量数据
            vector_data = self._generate_vectors_for_skills(list(set(all_role_skills)))

            role_baselines[role] = {
                'semantic_data': semantic_data,
                'vector_data': vector_data
            }

        return role_baselines

    def _generate_industry_baselines(self, industry_classifications: Dict) -> Dict:
        """生成行业基线"""
        logger.info("🏭 生成行业基线...")

        industry_baselines = {}

        for industry, categories in industry_classifications.items():
            all_industry_skills = []
            for skills in categories.values():
                all_industry_skills.extend(skills)

            if len(all_industry_skills) < self.config.min_skills_for_baseline:
                logger.warning(f"行业 {industry} 技能数量不足，跳过基线生成")
                continue

            # 语义数据（人类可读）
            semantic_data = {
                'industry_name': industry,
                'categories': categories,
                'total_skills': len(all_industry_skills),
                'skills_list': sorted(list(set(all_industry_skills))),
                'statistics': {
                    'category_distribution': {cat: len(skills) for cat, skills in categories.items() if skills}
                },
                'metadata': {
                    'generation_date': datetime.now().isoformat(),
                    'classification_model': self.config.classification_model,
                    'vector_model': 'all-MiniLM-L6-v2'
                }
            }

            # 向量数据
            vector_data = self._generate_vectors_for_skills(list(set(all_industry_skills)))

            industry_baselines[industry] = {
                'semantic_data': semantic_data,
                'vector_data': vector_data
            }

        return industry_baselines

    def _generate_vectors_for_skills(self, skills: List[str]) -> Dict:
        """为技能列表生成符合BaselineVectorSet schema的向量数据字典"""
        if not skills:
            return {}

        unique_skills = sorted(list(set(skills))) # 保证顺序稳定

        try:
            skill_embeddings = self.vector_model.encode(unique_skills)
            
            # 创建符合BaselineVectorSet schema的字典
            vector_set = BaselineVectorSet(
                vectors=[embedding.tolist() for embedding in skill_embeddings],
                skill_map={skill: i for i, skill in enumerate(unique_skills)},
                model='all-MiniLM-L6-v2',
                dimension=384
            )
            
            # Pydantic模型转为字典以便JSON序列化
            return vector_set.dict()
            
        except Exception as e:
            logger.error(f"向量生成失败: {e}")
            return {}

    def _generate_vectors_for_role(self, categories: Dict) -> Dict:
        """DEPRECATED: 向量生成已统一到 _generate_vectors_for_skills"""
        pass

    def _generate_vectors_for_industry(self, categories: Dict) -> Dict:
        """DEPRECATED: 向量生成已统一到 _generate_vectors_for_skills"""
        pass 