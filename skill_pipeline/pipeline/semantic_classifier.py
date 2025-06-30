import logging
from typing import Dict, List, Tuple, Optional
from .semantic_description_engine import ConfigurableSemanticEngine
from sentence_transformers import util

logger = logging.getLogger(__name__)

class SemanticClassifier:
    """基于配置驱动的增强语义分类器"""
    
    def __init__(self, config_dir: str):
        self.semantic_engine = ConfigurableSemanticEngine(config_dir)
        self.confidence_thresholds = self._load_confidence_thresholds()
        # 定义显式分配的权重
        self.explicit_weights = {
            'role': {'essential': 1.0, 'common': 0.85, 'specializations': 0.75, 'collaboration': 0.6},
            'industry': {'core_domain': 1.0, 'regulatory': 0.9, 'business_focus': 0.8, 'unique_requirements': 0.75, 'common': 0.6}
        }
        
    def _load_confidence_thresholds(self) -> Dict[str, float]:
        """加载置信度阈值"""
        thresholds = self.semantic_engine.engine_config.get('semantic_engine', {}).get('confidence_thresholds', {})
        return {
            'high': thresholds.get('high', 0.75),
            'medium': thresholds.get('medium', 0.55),
            'low': thresholds.get('low', 0.35)
        }
    
    def classify_skills(self, all_skills: List[Dict]) -> Tuple[Dict, Dict]:
        """
        执行加权多重分配技能分类。
        - 移除排他性：一个技能可以被分配给多个角色/行业。
        - 引入权重：每个分配都有一个权重，来源于显式定义或语义相似度。
        """
        logger.info("🧠 开始执行加权多重分配技能分类...")

        # 初始化分类结构
        role_classifications = {role: {cat: [] for cat in self.explicit_weights['role']} 
                              for role in self.semantic_engine.role_profiles.keys()}
        industry_classifications = {industry: {cat: [] for cat in self.explicit_weights['industry']}
                                  for industry in self.semantic_engine.industry_profiles.keys()}

        # 预先编码所有技能以提高效率
        skill_texts = [s['skill'] for s in all_skills]
        logger.info(f"正在分析 {len(skill_texts)} 个技能...")
        skill_embeddings = self.semantic_engine.model.encode(skill_texts, convert_to_tensor=True)
        
        # 计算所有技能与所有角色的相似度矩阵
        role_sim_matrix = util.pytorch_cos_sim(skill_embeddings, self.semantic_engine.role_embeddings)
        # 计算所有技能与所有行业的相似度矩阵
        industry_sim_matrix = util.pytorch_cos_sim(skill_embeddings, self.semantic_engine.industry_embeddings)
        
        role_names = list(self.semantic_engine.role_profiles.keys())
        industry_names = list(self.semantic_engine.industry_profiles.keys())

        for i, skill_info in enumerate(all_skills):
            skill = skill_info['skill']
            skill_lower = skill.lower()

            # --- 角色分配 ---
            for j, role_name in enumerate(role_names):
                profile = self.semantic_engine.role_profiles[role_name]
                
                # 1. 显式分配
                is_explicitly_assigned = False
                for category, explicit_skills in profile.explicit_skills.items():
                    if category in role_classifications[role_name] and (skill in explicit_skills or skill_lower in [s.lower() for s in explicit_skills]):
                        weight = self.explicit_weights['role'].get(category, 0.7)
                        role_classifications[role_name][category].append(
                            {'skill': skill, 'weight': weight, 'source': 'explicit'}
                        )
                        is_explicitly_assigned = True
                        break
                
                if is_explicitly_assigned:
                    continue

                # 2. 语义分配
                similarity = role_sim_matrix[i, j].item()
                if similarity >= self.confidence_thresholds['low']:
                    category = self.semantic_engine.determine_category(skill_info, similarity, 'role')
                    if category in role_classifications[role_name]:
                        role_classifications[role_name][category].append(
                            {'skill': skill, 'weight': round(similarity, 3), 'source': 'semantic'}
                        )

            # --- 行业分配 ---
            for k, industry_name in enumerate(industry_names):
                profile = self.semantic_engine.industry_profiles[industry_name]

                # 1. 显式分配
                is_explicitly_assigned = False
                for category, explicit_skills in profile.explicit_skills.items():
                    if category in industry_classifications[industry_name] and (skill in explicit_skills or skill_lower in [s.lower() for s in explicit_skills]):
                        weight = self.explicit_weights['industry'].get(category, 0.7)
                        industry_classifications[industry_name][category].append(
                            {'skill': skill, 'weight': weight, 'source': 'explicit'}
                        )
                        is_explicitly_assigned = True
                        break
                
                if is_explicitly_assigned:
                    continue
                    
                # 2. 语义分配
                similarity = industry_sim_matrix[i, k].item()
                if similarity >= self.confidence_thresholds['low']:
                    category = self.semantic_engine.determine_category(skill_info, similarity, 'industry')
                    if category in industry_classifications[industry_name]:
                        industry_classifications[industry_name][category].append(
                            {'skill': skill, 'weight': round(similarity, 3), 'source': 'semantic'}
                        )

        # 清理并返回结果
        role_classifications = self._clean_classifications(role_classifications)
        industry_classifications = self._clean_classifications(industry_classifications)
        
        logger.info("✅ 加权多重分配分类完成.")
        return role_classifications, industry_classifications

    def _clean_classifications(self, classifications: Dict) -> Dict:
        """清理分类结果，移除空列表并按技能名去重"""
        cleaned = {}
        for key, categories in classifications.items():
            cleaned_categories = {}
            for category_name, skills in categories.items():
                if isinstance(skills, list) and skills:
                    # 按技能名称去重，保留权重最高的一个
                    skill_map = {}
                    for s in skills:
                        if s['skill'] not in skill_map or s['weight'] > skill_map[s['skill']]['weight']:
                            skill_map[s['skill']] = s
                    
                    unique_skills = list(skill_map.values())
                    # 按权重降序排序
                    cleaned_categories[category_name] = sorted(unique_skills, key=lambda x: x['weight'], reverse=True)

            if cleaned_categories:
                cleaned[key] = cleaned_categories
        return cleaned
    