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

        all_skills_with_weights = []
        for categories in role_classifications.values():
            for skills in categories.values():
                all_skills_with_weights.extend(skills)

        for categories in industry_classifications.values():
            for skills in categories.values():
                all_skills_with_weights.extend(skills)

        # 按技能名称去重，保留最高权重
        skill_map = {}
        for s in all_skills_with_weights:
            if s['skill'] not in skill_map or s['weight'] > skill_map[s['skill']]['weight']:
                skill_map[s['skill']] = s
        
        unique_skills_with_weights = sorted(list(skill_map.values()), key=lambda x: x['skill'])
        all_skills = [s['skill'] for s in unique_skills_with_weights]

        # 统计信息
        role_distribution = {role: sum(len(skills) for skills in categories.values())
                             for role, categories in role_classifications.items()}
        industry_distribution = {industry: sum(len(skills) for skills in categories.values())
                                 for industry, categories in industry_classifications.items()}

        # 分离语义数据和向量数据
        semantic_data = {
            'total_skills': len(all_skills),
            'skills_list': unique_skills_with_weights,  # 技能列表（包含权重）
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
        vector_data = self._generate_vectors_for_skills(unique_skills_with_weights)

        return {
            'semantic_data': semantic_data,
            'vector_data': vector_data
        }

    def _generate_role_baselines(self, role_classifications: Dict) -> Dict:
        """生成角色基线"""
        logger.info("👔 生成角色基线...")

        role_baselines = {}

        for role, categories in role_classifications.items():
            all_role_skills_with_weights = []
            for skills in categories.values():
                all_role_skills_with_weights.extend(skills)
            
            # 按技能名去重，保留最高权重 (因为一个技能可能通过显式和语义两种方式分配到同一个角色)
            skill_map = {}
            for s in all_role_skills_with_weights:
                if s['skill'] not in skill_map or s['weight'] > skill_map[s['skill']]['weight']:
                    skill_map[s['skill']] = s
            
            unique_skills_with_weights = sorted(list(skill_map.values()), key=lambda x: x['weight'], reverse=True)

            if len(unique_skills_with_weights) < self.config.min_skills_for_baseline:
                logger.warning(f"角色 {role} 技能数量不足 ({len(unique_skills_with_weights)}), 跳过基线生成")
                continue

            # 语义数据（人类可读）
            semantic_data = {
                'role_name': role,
                'categories': categories, # categories already contain weights
                'total_skills': len(unique_skills_with_weights),
                'skills_list': unique_skills_with_weights,
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
            vector_data = self._generate_vectors_for_skills(unique_skills_with_weights)

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
            all_industry_skills_with_weights = []
            for skills in categories.values():
                all_industry_skills_with_weights.extend(skills)

            # 按技能名去重
            skill_map = {}
            for s in all_industry_skills_with_weights:
                if s['skill'] not in skill_map or s['weight'] > skill_map[s['skill']]['weight']:
                    skill_map[s['skill']] = s
            unique_skills_with_weights = sorted(list(skill_map.values()), key=lambda x: x['weight'], reverse=True)


            if len(unique_skills_with_weights) < self.config.min_skills_for_baseline:
                logger.warning(f"行业 {industry} 技能数量不足 ({len(unique_skills_with_weights)}), 跳过基线生成")
                continue

            # 语义数据（人类可读）
            semantic_data = {
                'industry_name': industry,
                'categories': categories,
                'total_skills': len(unique_skills_with_weights),
                'skills_list': unique_skills_with_weights,
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
            vector_data = self._generate_vectors_for_skills(unique_skills_with_weights)

            industry_baselines[industry] = {
                'semantic_data': semantic_data,
                'vector_data': vector_data
            }

        return industry_baselines

    def _generate_vectors_for_skills(self, skills_with_weights: List[Dict]) -> Dict:
        """
        为技能列表生成符合BaselineVectorSet schema的向量数据字典。
        现在接收一个包含技能和权重的字典列表，并计算加权平均向量。
        """
        if not skills_with_weights:
            return {}

        skills = [item['skill'] for item in skills_with_weights]
        weights = np.array([item['weight'] for item in skills_with_weights])

        try:
            skill_embeddings = self.vector_model.encode(skills)
            
            # 计算加权平均向量
            weighted_average_vector = np.average(skill_embeddings, axis=0, weights=weights)

            # 创建符合BaselineVectorSet schema的字典
            # 注意：'vectors' 字段现在存储的是单个加权平均向量
            vector_set = BaselineVectorSet(
                vectors=[weighted_average_vector.tolist()],  # 这是一个列表，包含一个向量
                skill_map={skill: i for i, skill in enumerate(skills)}, # skill_map 保持不变
                model='all-MiniLM-L6-v2_weighted_avg', # 更新模型名称以反映变化
                dimension=384,
                # 可以选择性地添加权重信息
                weights={item['skill']: item['weight'] for item in skills_with_weights}
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