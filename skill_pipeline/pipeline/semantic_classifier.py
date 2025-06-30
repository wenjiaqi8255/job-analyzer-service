import logging
from typing import Dict, List, Tuple, Optional
from .semantic_description_engine import ConfigurableSemanticEngine

logger = logging.getLogger(__name__)

class SemanticClassifier:
    """基于配置驱动的增强语义分类器"""
    
    def __init__(self, config_dir: str):
        self.semantic_engine = ConfigurableSemanticEngine(config_dir)
        self.confidence_thresholds = self._load_confidence_thresholds()
        
    def _load_confidence_thresholds(self) -> Dict[str, float]:
        """加载置信度阈值"""
        thresholds = self.semantic_engine.engine_config.get('semantic_engine', {}).get('confidence_thresholds', {})
        return {
            'high': thresholds.get('high', 0.75),
            'medium': thresholds.get('medium', 0.55),
            'low': thresholds.get('low', 0.35)
        }
    
    def classify_skills(self, all_skills: List[Dict]) -> Tuple[Dict, Dict]:
        """分级分类所有技能"""
        logger.info("🧠 开始配置驱动的技能分类...")
        
        # 初始化分类结构
        role_classifications = {role: {
            'essential': [], 'common': [], 'collaboration': [], 'specializations': []
        } for role in self.semantic_engine.role_profiles.keys()}
        
        industry_classifications = {industry: {
            'core_domain': [], 'regulatory': [], 'business_focus': [], 'unique_requirements': [], 'common': []
        } for industry in self.semantic_engine.industry_profiles.keys()}
        
        # 统计变量
        stats = {
            'explicit_role_assigned': 0,
            'explicit_industry_assigned': 0,
            'semantic_role_assigned': 0,
            'semantic_industry_assigned': 0,
            'unassigned': 0,
            'skipped_by_exclusion': 0
        }
        
        assigned_skills = set()
        
        # === 第一轮：显式规则分配 ===
        logger.info("📋 第一轮：显式规则分配...")
        for skill_info in all_skills:
            skill = skill_info['skill']
            
            # 角色显式分配
            role_assigned = self._assign_explicit_role(skill, role_classifications)
            if role_assigned:
                assigned_skills.add(skill)
                stats['explicit_role_assigned'] += 1
                continue
            
            # 行业显式分配
            industry_assigned = self._assign_explicit_industry(skill, industry_classifications)
            if industry_assigned:
                assigned_skills.add(skill)
                stats['explicit_industry_assigned'] += 1
        
        # === 第二轮：高置信度语义匹配 ===
        logger.info("🎯 第二轮：高置信度语义匹配...")
        self._semantic_assignment_round(all_skills, assigned_skills, role_classifications, 
                                      industry_classifications, stats, 'high')
        
        # === 第三轮：中等置信度语义匹配 ===
        logger.info("🔍 第三轮：中等置信度语义匹配...")
        self._semantic_assignment_round(all_skills, assigned_skills, role_classifications, 
                                      industry_classifications, stats, 'medium')
        
        # === 第四轮：兜底处理 ===
        logger.info("🥅 第四轮：兜底处理...")
        self._fallback_assignment(all_skills, assigned_skills, role_classifications, 
                                industry_classifications, stats)
        
        # 清理空分类
        role_classifications = self._clean_classifications(role_classifications)
        industry_classifications = self._clean_classifications(industry_classifications)
        
        # 输出统计信息
        self._log_classification_stats(stats, len(all_skills))
        
        return role_classifications, industry_classifications
    
    def _assign_explicit_role(self, skill: str, role_classifications: Dict) -> bool:
        """显式角色分配 - 基于配置文件"""
        skill_lower = skill.lower()
        
        for role_name, profile in self.semantic_engine.role_profiles.items():
            for category, skills in profile.explicit_skills.items():
                if skill in skills or skill_lower in [s.lower() for s in skills]:
                    role_classifications[role_name][category].append(skill)
                    logger.debug(f"  📋 显式分配: {skill} → {role_name}.{category}")
                    return True
        return False
    
    def _assign_explicit_industry(self, skill: str, industry_classifications: Dict) -> bool:
        """显式行业分配 - 基于配置文件"""
        skill_lower = skill.lower()
        
        for industry_name, profile in self.semantic_engine.industry_profiles.items():
            for category, skills in profile.explicit_skills.items():
                if skill in skills or skill_lower in [s.lower() for s in skills]:
                    industry_classifications[industry_name][category].append(skill)
                    logger.debug(f"  🏢 显式分配: {skill} → {industry_name}.{category}")
                    return True
        return False
    
    def _semantic_assignment_round(self, all_skills: List[Dict], assigned_skills: set,
                                 role_classifications: Dict, industry_classifications: Dict,
                                 stats: Dict, confidence_level: str):
        """执行一轮语义分配"""
        threshold = self.confidence_thresholds[confidence_level]
        
        for skill_info in all_skills:
            skill = skill_info['skill']
            
            if skill in assigned_skills:
                continue
            
            # 获取最佳角色匹配
            best_role, role_similarity, role_profile = self.semantic_engine.get_best_role_match(skill_info)
            
            if best_role and role_similarity >= threshold:
                category = self.semantic_engine.determine_category(skill_info, role_similarity, 'role')
                role_classifications[best_role][category].append(skill)
                assigned_skills.add(skill)
                stats['semantic_role_assigned'] += 1
                logger.debug(f"  🎯 语义角色分配: {skill} → {best_role}.{category} (相似度: {role_similarity:.3f})")
                continue
            
            # 获取最佳行业匹配
            best_industry, industry_similarity, industry_profile = self.semantic_engine.get_best_industry_match(skill_info)
            
            if best_industry and industry_similarity >= threshold:
                category = self.semantic_engine.determine_category(skill_info, industry_similarity, 'industry')
                industry_classifications[best_industry][category].append(skill)
                assigned_skills.add(skill)
                stats['semantic_industry_assigned'] += 1
                logger.debug(f"  🏭 语义行业分配: {skill} → {best_industry}.{category} (相似度: {industry_similarity:.3f})")
    
    def _fallback_assignment(self, all_skills: List[Dict], assigned_skills: set,
                           role_classifications: Dict, industry_classifications: Dict,
                           stats: Dict):
        """兜底分配处理"""
        for skill_info in all_skills:
            skill = skill_info['skill']
            
            if skill in assigned_skills:
                continue
            
            # 尝试低置信度匹配
            best_role, role_similarity, _ = self.semantic_engine.get_best_role_match(skill_info)
            best_industry, industry_similarity, _ = self.semantic_engine.get_best_industry_match(skill_info)
            
            low_threshold = self.confidence_thresholds['low']
            
            # 选择更好的匹配
            if (best_role and role_similarity >= low_threshold and 
                role_similarity > industry_similarity):
                category = self.semantic_engine.determine_category(skill_info, role_similarity, 'role')
                role_classifications[best_role][category].append(skill)
                stats['semantic_role_assigned'] += 1
            elif best_industry and industry_similarity >= low_threshold:
                category = self.semantic_engine.determine_category(skill_info, industry_similarity, 'industry')
                industry_classifications[best_industry][category].append(skill)
                stats['semantic_industry_assigned'] += 1
            else:
                # 最终兜底到tech行业
                if 'tech' in industry_classifications:
                    industry_classifications['tech']['core_domain'].append(skill)
                stats['unassigned'] += 1
    
    def _clean_classifications(self, classifications: Dict) -> Dict:
        """清理分类结果"""
        cleaned = {}
        for key, categories in classifications.items():
            cleaned_categories = {}
            for category_name, skills in categories.items():
                if isinstance(skills, list) and skills:
                    unique_skills = sorted(list(set(skills)))
                    cleaned_categories[category_name] = unique_skills
            if cleaned_categories:
                cleaned[key] = cleaned_categories
        return cleaned
    
    def _log_classification_stats(self, stats: Dict, total_skills: int):
        """记录分类统计信息"""
        logger.info(f"📊 分类完成:")
        logger.info(f"  📋 显式角色分配: {stats['explicit_role_assigned']}")
        logger.info(f"  🏢 显式行业分配: {stats['explicit_industry_assigned']}")
        logger.info(f"  🎯 语义角色分配: {stats['semantic_role_assigned']}")
        logger.info(f"  🏭 语义行业分配: {stats['semantic_industry_assigned']}")
        logger.info(f"  🥅 兜底分配: {stats['unassigned']}")
        logger.info(f"  📈 总处理: {total_skills} 个技能")
    