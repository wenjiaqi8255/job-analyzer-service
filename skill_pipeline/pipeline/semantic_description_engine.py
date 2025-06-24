import yaml
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class SemanticProfile:
    """语义描述配置文件"""
    name: str
    semantic_description: str
    explicit_skills: Dict[str, List[str]]
    skill_type_preferences: Dict[str, float]
    exclusion_rules: Dict[str, List[str]]
    alternative_names: List[str]
    domain_keywords: List[str]
    related_entities: List[str]

class ConfigurableSemanticEngine:
    """配置驱动的语义引擎 - 只添加缓存优化"""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.model = None
        self.role_profiles = {}
        self.industry_profiles = {}
        self.engine_config = {}
        
        # 新增：预计算的profile embeddings缓存
        self._profile_embeddings_cache = {}
        
        self._load_all_configs()
        self._initialize_model()
        
        # 核心优化：预计算所有profile embeddings
        self._precompute_profile_embeddings()
    
    def _precompute_profile_embeddings(self):
        """预计算所有profile的embeddings - 核心性能优化"""
        logger.info("🔥 预计算profile embeddings...")
        
        # 收集所有profile描述
        all_descriptions = []
        profile_keys = []
        
        # 角色profiles
        for role_name, profile in self.role_profiles.items():
            all_descriptions.append(profile.semantic_description)
            profile_keys.append(f"role_{role_name}")
        
        # 行业profiles  
        for industry_name, profile in self.industry_profiles.items():
            all_descriptions.append(profile.semantic_description)
            profile_keys.append(f"industry_{industry_name}")
        
        # 批量计算所有embeddings
        if all_descriptions:
            embeddings = self.model.encode(
                all_descriptions, 
                batch_size=32,
                show_progress_bar=True
            )
            
            # 存储到缓存
            for key, embedding in zip(profile_keys, embeddings):
                self._profile_embeddings_cache[key] = embedding
        
        logger.info(f"✅ 预计算完成: {len(all_descriptions)} 个profile embeddings")
    
    def _load_all_configs(self):
        """加载所有配置文件"""
        # 加载角色配置
        role_config_path = self.config_dir / "role_definitions.yaml"
        if role_config_path.exists():
            with open(role_config_path, 'r', encoding='utf-8') as f:
                role_data = yaml.safe_load(f)
                self.role_profiles = self._parse_profiles(role_data.get('roles', {}), 'role')
        
        # 加载行业配置
        industry_config_path = self.config_dir / "industry_definitions.yaml"
        if industry_config_path.exists():
            with open(industry_config_path, 'r', encoding='utf-8') as f:
                industry_data = yaml.safe_load(f)
                self.industry_profiles = self._parse_profiles(industry_data.get('industries', {}), 'industry')
        
        # 加载引擎配置
        engine_config_path = self.config_dir / "semantic_engine_config.yaml"
        if engine_config_path.exists():
            with open(engine_config_path, 'r', encoding='utf-8') as f:
                self.engine_config = yaml.safe_load(f)
        
        logger.info(f"✅ 加载配置: {len(self.role_profiles)} 角色, {len(self.industry_profiles)} 行业")
    
    def _parse_profiles(self, config_data: Dict, profile_type: str) -> Dict[str, SemanticProfile]:
        """解析配置数据为语义配置文件"""
        profiles = {}
        
        for name, data in config_data.items():
            profiles[name] = SemanticProfile(
                name=name,
                semantic_description=data.get('semantic_description', ''),
                explicit_skills=data.get('explicit_skills', {}),
                skill_type_preferences=data.get('skill_type_preferences', {}),
                exclusion_rules=data.get('exclusion_rules', {}),
                alternative_names=data.get('alternative_names', []),
                domain_keywords=data.get('domain_keywords', []),
                related_entities=data.get(f'related_{profile_type}s', [])
            )
        
        return profiles
    
    def _initialize_model(self):
        """初始化embedding模型"""
        model_name = self.engine_config.get('semantic_engine', {}).get('embedding_model', 
                                                                       'sentence-transformers/all-MiniLM-L6-v2')
        backup_model = self.engine_config.get('semantic_engine', {}).get('backup_model', 
                                                                         'sentence-transformers/paraphrase-MiniLM-L6-v2')
        
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ 成功加载模型: {model_name}")
        except Exception as e:
            logger.warning(f"⚠️ 主模型加载失败，使用备选模型: {e}")
            self.model = SentenceTransformer(backup_model)
    
    def enhance_skill_description(self, skill_info: Dict, target_profile: Optional[SemanticProfile] = None) -> str:
        """增强技能描述 - 使用配置文件中的信息"""
        skill = skill_info['skill']
        skill_type = skill_info['skill_type']
        context = skill_info.get('context', '')
        
        # 基础描述
        base_description = skill.replace('_', ' ').replace('-', ' ')
        
        # 如果指定了目标配置文件，使用其domain keywords
        enhanced_parts = [base_description]
        
        if target_profile and target_profile.domain_keywords:
            relevant_keywords = [kw for kw in target_profile.domain_keywords 
                               if kw.lower() in skill.lower() or skill.lower() in kw.lower()]
            enhanced_parts.extend(relevant_keywords)
        
        # 技能类型增强
        type_enhancements = {
            'programming_language': 'programming language software development coding',
            'framework': 'framework web development software engineering',
            'database': 'database management data storage query optimization',
            'cloud_service': 'cloud computing infrastructure deployment scalability',
            'tool': 'tool software productivity development',
            'methodology': 'methodology approach process management',
            'certification': 'certification professional qualification expertise',
            'language': 'language communication international business',
            'domain_knowledge': f'domain expertise industry knowledge {context}'
        }
        
        if skill_type in type_enhancements:
            enhanced_parts.append(type_enhancements[skill_type])
        
        return ' '.join(enhanced_parts)
    
    @lru_cache(maxsize=5000)
    def _get_skill_embedding_cached(self, enhanced_skill: str):
        """缓存版本的技能embedding计算"""
        return self.model.encode([enhanced_skill])[0]
    
    def calculate_semantic_similarity(self, skill_info: Dict, profile: SemanticProfile) -> float:
        """计算语义相似度 - 使用缓存优化"""
        enhanced_skill = self.enhance_skill_description(skill_info, profile)
        
        try:
            # 1. 获取技能embedding - 使用缓存
            skill_embedding = self._get_skill_embedding_cached(enhanced_skill)
            
            # 2. 获取profile embedding - 使用预计算的缓存
            profile_key = self._get_profile_cache_key(profile)
            profile_embedding = self._profile_embeddings_cache.get(profile_key)
            
            if profile_embedding is None:
                logger.warning(f"⚠️ Profile embedding not found for: {profile_key}")
                # 临时计算（不应该发生）
                profile_embedding = self.model.encode([profile.semantic_description])[0]
            
            # 3. 计算余弦相似度 - 保持原有方式
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([skill_embedding], [profile_embedding])[0][0]
            
            # 4. 应用权重调整
            adjusted_similarity = self._apply_similarity_bonuses(skill_info, profile, similarity)
            
            return adjusted_similarity
            
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return 0.0
    
    def _get_profile_cache_key(self, profile: SemanticProfile) -> str:
        """获取profile的缓存key"""
        if profile.name in self.role_profiles:
            return f"role_{profile.name}"
        elif profile.name in self.industry_profiles:
            return f"industry_{profile.name}"
        else:
            return f"unknown_{profile.name}"
    
    def _apply_similarity_bonuses(self, skill_info: Dict, profile: SemanticProfile, base_similarity: float) -> float:
        """应用相似度加分"""
        skill = skill_info['skill'].lower()
        skill_type = skill_info['skill_type']
        
        bonus = 0.0
        bonus_config = self.engine_config.get('semantic_engine', {}).get('bonus_weights', {})
        
        # 技能类型偏好加分
        if skill_type in profile.skill_type_preferences:
            preference = profile.skill_type_preferences[skill_type]
            type_bonus = bonus_config.get('skill_type_match', 0.15) * preference
            bonus += type_bonus
        
        # 显式技能匹配加分
        for category, skills in profile.explicit_skills.items():
            if skill in [s.lower() for s in skills]:
                bonus += bonus_config.get('explicit_keyword_match', 0.25)
                break
        
        # 替代名称匹配
        for alt_name in profile.alternative_names:
            if alt_name.lower() in skill or skill in alt_name.lower():
                bonus += bonus_config.get('explicit_keyword_match', 0.25) * 0.8
                break
        
        return base_similarity * (1 + bonus)
    
    def get_best_role_match(self, skill_info: Dict) -> Tuple[Optional[str], float, SemanticProfile]:
        """获取最佳角色匹配"""
        best_role = None
        best_similarity = 0.0
        best_profile = None
        
        for role_name, profile in self.role_profiles.items():
            # 检查排除规则
            if self._is_excluded_by_profile(skill_info, profile):
                continue
                
            similarity = self.calculate_semantic_similarity(skill_info, profile)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_role = role_name
                best_profile = profile
        
        return best_role, best_similarity, best_profile
    
    def get_best_industry_match(self, skill_info: Dict) -> Tuple[Optional[str], float, SemanticProfile]:
        """获取最佳行业匹配"""
        best_industry = None
        best_similarity = 0.0
        best_profile = None
        
        for industry_name, profile in self.industry_profiles.items():
            similarity = self.calculate_semantic_similarity(skill_info, profile)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_industry = industry_name
                best_profile = profile
        
        return best_industry, best_similarity, best_profile
    
    def _is_excluded_by_profile(self, skill_info: Dict, profile: SemanticProfile) -> bool:
        """检查是否被配置文件排除"""
        skill = skill_info['skill']
        skill_type = skill_info['skill_type']
        
        excluded_skills = profile.exclusion_rules.get('skills', [])
        excluded_categories = profile.exclusion_rules.get('categories', [])
        
        return skill in excluded_skills or skill_type in excluded_categories
    
    def determine_category(self, skill_info: Dict, similarity: float, profile_type: str = 'role') -> str:
        """根据相似度确定技能类别"""
        thresholds = self.engine_config.get('semantic_engine', {}).get('category_assignment_rules', {})
        skill_type = skill_info['skill_type']
        
        essential_threshold = thresholds.get('essential_threshold', 0.85)
        common_threshold = thresholds.get('common_threshold', 0.75)
        specialization_threshold = thresholds.get('specialization_threshold', 0.65)
        
        if similarity >= essential_threshold:
            return 'essential'
        elif similarity >= common_threshold:
            # 根据技能类型决定是essential还是common
            if skill_type in ['programming_language', 'framework', 'database']:
                return 'essential'
            else:
                return 'common'
        elif similarity >= specialization_threshold:
            if skill_type in ['certification', 'methodology']:
                return 'specializations'
            else:
                return 'common'
        else:
            return 'collaboration' if profile_type == 'role' else 'unique_requirements'
    
    def add_role_profile(self, role_name: str, profile_data: Dict):
        """动态添加角色配置文件"""
        self.role_profiles[role_name] = SemanticProfile(
            name=role_name,
            **profile_data
        )
        
        # 计算新profile的embedding并加入缓存
        profile_embedding = self.model.encode([profile_data.get('semantic_description', '')])[0]
        self._profile_embeddings_cache[f"role_{role_name}"] = profile_embedding
        
        logger.info(f"✅ 添加角色配置: {role_name}")
    
    def add_industry_profile(self, industry_name: str, profile_data: Dict):
        """动态添加行业配置文件"""
        self.industry_profiles[industry_name] = SemanticProfile(
            name=industry_name,
            **profile_data
        )
        
        # 计算新profile的embedding并加入缓存
        profile_embedding = self.model.encode([profile_data.get('semantic_description', '')])[0]
        self._profile_embeddings_cache[f"industry_{industry_name}"] = profile_embedding
        
        logger.info(f"✅ 添加行业配置: {industry_name}")
    
    def export_config_template(self, output_path: str):
        """导出配置模板"""
        template = {
            'roles': {
                'example_role': {
                    'semantic_description': 'Detailed description of the role responsibilities and skills',
                    'explicit_skills': {
                        'essential': ['skill1', 'skill2'],
                        'common': ['skill3', 'skill4']
                    },
                    'skill_type_preferences': {
                        'programming_language': 0.8,
                        'tool': 0.6
                    },
                    'exclusion_rules': {
                        'skills': ['excluded_skill'],
                        'categories': ['excluded_category']
                    },
                    'alternative_names': ['alt_name1', 'alt_name2'],
                    'domain_keywords': ['keyword1', 'keyword2'],
                    'related_industries': ['industry1', 'industry2']
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(template, f, default_flow_style=False, allow_unicode=True)