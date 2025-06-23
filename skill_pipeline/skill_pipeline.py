#!/usr/bin/env python3
"""
JobbAI 端到端技能处理管线
从vocabulary到三个baseline的完整流程
"""

import json
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict, Counter

# 第三方库
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

class SkillProcessor:
    """技能预处理和标准化"""
    
    def __init__(self):
        self.skill_types = {
            'programming_language': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'scala', 'kotlin', 'swift'],
            'framework': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'laravel', 'express'],
            'database': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra'],
            'cloud_service': ['aws', 'azure', 'google_cloud', 'docker', 'kubernetes'],
            'tool': ['figma', 'sketch', 'photoshop', 'tableau', 'jira', 'confluence'],
            'methodology': ['agile', 'scrum', 'lean', 'kanban', 'devops'],
            'certification': ['certified', 'certification', 'associate', 'professional'],
            'language': ['english', 'german', 'french', 'spanish', 'chinese', 'japanese']
        }
    
    def extract_all_skills(self, skills_vocab: Dict) -> List[Dict]:
        """从技能词汇中提取所有技能"""
        all_skills = []
        
        def extract_recursive(data, path="", context=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    new_context = f"{context} {key}".strip()
                    if isinstance(value, list):
                        for skill in value:
                            all_skills.append({
                                'skill': skill,
                                'path': new_path,
                                'context': new_context,
                                'skill_type': self._infer_skill_type(skill, key)
                            })
                    else:
                        extract_recursive(value, new_path, new_context)
        
        extract_recursive(skills_vocab)
        return all_skills
    
    def _infer_skill_type(self, skill: str, category: str) -> str:
        """推断技能类型"""
        skill_lower = skill.lower()
        category_lower = category.lower()
        
        for skill_type, keywords in self.skill_types.items():
            if any(keyword in skill_lower for keyword in keywords) or skill_type in category_lower:
                return skill_type
        
        return 'domain_knowledge'

class SemanticClassifier:
    """基于显式规则 + 语义补充的智能分类器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = self._load_model()
        self._load_semantic_descriptions()
        self._load_explicit_mappings()
        
        # 分级阈值
        self.high_confidence_threshold = getattr(config, 'high_confidence_threshold', 0.75)
        self.medium_confidence_threshold = getattr(config, 'medium_confidence_threshold', 0.55)
        self.low_confidence_threshold = getattr(config, 'low_confidence_threshold', 0.35)
        
    def _load_model(self) -> SentenceTransformer:
        """加载SentenceTransformer模型"""
        logger.info("🤖 加载SentenceTransformer模型...")
        try:
            model = SentenceTransformer(self.config.classification_model)
            logger.info(f"✅ 成功加载专用模型: {self.config.classification_model}")
            return model
        except Exception as e:
            logger.warning(f"⚠️ 无法加载专用模型，使用备选模型: {e}")
            model = SentenceTransformer(self.config.backup_model)
            logger.info(f"✅ 成功加载备选模型: {self.config.backup_model}")
            return model
    
    def _assign_explicit_role(self, skill: str, role_classifications: Dict) -> bool:
        """显式角色分配"""
        skill_lower = skill.lower()
        
        for role, categories in self.explicit_role_mappings.items():
            for category, skills in categories.items():
                if skill in skills or skill_lower in skills:
                    role_classifications[role][category].append(skill)
                    logger.debug(f"  📋 显式分配: {skill} → {role}.{category}")
                    return True
        return False
    
    def _assign_explicit_industry(self, skill: str, industry_classifications: Dict) -> bool:
        """显式行业分配"""
        skill_lower = skill.lower()
        
        for industry, categories in self.explicit_industry_mappings.items():
            for category, skills in categories.items():
                if skill in skills or skill_lower in skills:
                    industry_classifications[industry][category].append(skill)
                    logger.debug(f"  🏢 显式分配: {skill} → {industry}.{category}")
                    return True
        return False
    
    def _is_excluded_by_rules(self, skill_info: Dict) -> bool:
        """检查技能是否被排除规则排除"""
        skill = skill_info['skill']
        skill_type = skill_info['skill_type']
        
        for role, rules in self.exclusion_rules.items():
            exclude_skills = rules.get('exclude_skills', [])
            exclude_categories = rules.get('exclude_categories', [])
            
            if skill in exclude_skills or skill_type in exclude_categories:
                logger.debug(f"  ❌ 排除规则: {skill} 不适合 {role}")
                return True
        
        return False
    
    def enhance_skill_description(self, skill_info: Dict) -> str:
        """增强技能描述"""
        skill = skill_info['skill']
        skill_type = skill_info['skill_type']
        context = skill_info['context']
        
        base_description = skill.replace('_', ' ').replace('-', ' ')
        
        enhancement_templates = {
            'programming_language': f'{base_description} programming language software development coding',
            'framework': f'{base_description} framework web development software engineering',
            'database': f'{base_description} database management data storage query optimization',
            'cloud_service': f'{base_description} cloud computing infrastructure deployment scalability',
            'tool': f'{base_description} tool software productivity development',
            'methodology': f'{base_description} methodology approach process management',
            'certification': f'{base_description} certification professional qualification expertise',
            'language': f'{base_description} language communication international business',
            'domain_knowledge': f'{base_description} {context} domain expertise industry knowledge'
        }
        
        return enhancement_templates.get(skill_type, f"{base_description} {context}")
    
    def _load_explicit_mappings(self):
        """加载显式技能映射规则"""
        logger.info("📋 加载显式技能映射规则...")
        
        # 核心显式技能映射 - 确保这些技能分配正确
        self.explicit_role_mappings = {
            'ui_ux_designer': {
                'essential': [
                    'figma', 'sketch', 'adobe_xd', 'adobe_photoshop', 'adobe_illustrator',
                    'invision', 'principle', 'framer', 'zeplin', 'marvel',
                    'user_interface_design', 'user_experience_design', 'ui_design', 'ux_design',
                    'wireframing', 'prototyping', 'user_research', 'usability_testing',
                    'interaction_design', 'visual_design', 'design_systems'
                ],
                'common': ['adobe_creative_suite', 'canva', 'miro', 'figjam']
            },
            'digital_marketing_specialist': {
                'essential': [
                    'google_ads', 'facebook_ads', 'google_analytics', 'social_media_marketing',
                    'seo', 'sem', 'content_marketing', 'email_marketing', 'instagram_marketing',
                    'linkedin_marketing', 'twitter_marketing', 'youtube_marketing',
                    'google_tag_manager', 'facebook_pixel', 'mailchimp', 'hubspot'
                ],
                'common': ['adobe_creative_suite', 'canva', 'hootsuite', 'buffer']
            },
            'project_manager': {
                'essential': [
                    'agile', 'scrum', 'project_management', 'jira', 'confluence',
                    'kanban', 'waterfall', 'risk_management', 'stakeholder_management',
                    'budget_management', 'timeline_management', 'resource_planning'
                ],
                'common': ['microsoft_project', 'asana', 'trello', 'monday_com', 'slack']
            },
            'data_scientist': {
                'essential': [
                    'python', 'r', 'sql', 'pandas', 'numpy', 'scikit_learn', 'tensorflow',
                    'pytorch', 'jupyter', 'matplotlib', 'seaborn', 'plotly',
                    'machine_learning', 'deep_learning', 'statistical_analysis',
                    'data_visualization', 'data_mining', 'predictive_modeling'
                ],
                'common': ['tableau', 'power_bi', 'excel', 'spark', 'hadoop']
            },
            'cybersecurity_specialist': {
                'essential': [
                    'penetration_testing', 'vulnerability_assessment', 'incident_response',
                    'security_auditing', 'threat_modeling', 'malware_analysis',
                    'network_security', 'web_application_security', 'cryptography',
                    'security_frameworks', 'iso_27001', 'nist_framework'
                ],
                'common': ['kali_linux', 'metasploit', 'wireshark', 'burp_suite', 'nmap']
            },
            'frontend_web_developer': {
                'essential': [
                    'html', 'css', 'javascript', 'react', 'angular', 'vue',
                    'typescript', 'sass', 'less', 'webpack', 'npm', 'yarn',
                    'responsive_design', 'cross_browser_compatibility'
                ],
                'common': ['figma', 'sketch', 'git', 'vscode']
            },
            'backend_developer': {
                'essential': [
                    'python', 'java', 'nodejs', 'c#', 'php', 'ruby', 'go',
                    'api_development', 'rest_api', 'graphql', 'microservices',
                    'database_design', 'sql', 'mongodb', 'postgresql', 'mysql'
                ],
                'common': ['docker', 'kubernetes', 'redis', 'elasticsearch']
            },
            'devops_engineer': {
                'essential': [
                    'docker', 'kubernetes', 'jenkins', 'gitlab_ci', 'github_actions',
                    'terraform', 'ansible', 'chef', 'puppet', 'aws', 'azure', 'gcp',
                    'continuous_integration', 'continuous_deployment', 'infrastructure_as_code'
                ],
                'common': ['linux', 'bash', 'python', 'monitoring', 'logging']
            },
            'cloud_engineer': {
                'essential': [
                    'aws', 'azure', 'google_cloud', 'docker', 'kubernetes',
                    'terraform', 'cloudformation', 'serverless', 'lambda',
                    'cloud_architecture', 'cloud_security', 'cost_optimization'
                ],
                'common': ['python', 'bash', 'networking', 'monitoring']
            }
        }
        
        # 显式行业映射
        self.explicit_industry_mappings = {
            'automotive': {
                'core_domain': [
                    'automotive_engineering', 'vehicle_dynamics', 'powertrain',
                    'electric_vehicles', 'autonomous_driving', 'adas',
                    'can_bus', 'autosar', 'iso_26262', 'automotive_safety'
                ]
            },
            'healthcare': {
                'core_domain': [
                    'clinical_research', 'medical_devices', 'pharmaceutical',
                    'biotechnology', 'healthcare_compliance', 'hipaa', 'fda_regulations',
                    'clinical_trials', 'medical_imaging', 'telemedicine'
                ]
            },
            'finance': {
                'core_domain': [
                    'financial_modeling', 'risk_management', 'trading', 'investment',
                    'banking', 'fintech', 'blockchain', 'cryptocurrency',
                    'regulatory_compliance', 'basel_iii', 'mifid_ii'
                ],
                'regulatory': ['kyc', 'aml', 'gdpr', 'pci_dss', 'sox_compliance']
            },
            'energy': {
                'core_domain': [
                    'renewable_energy', 'solar_energy', 'wind_energy', 'energy_storage',
                    'smart_grid', 'power_systems', 'electrical_engineering',
                    'energy_management', 'sustainability'
                ]
            },
            'aerospace': {
                'core_domain': [
                    'aerospace_engineering', 'avionics', 'flight_systems',
                    'satellite_technology', 'space_systems', 'do_178c',
                    'aviation_safety', 'flight_testing'
                ]
            }
        }
        
        # 技能排除规则 - 某些技能不应该分配给特定角色
        self.exclusion_rules = {
            'ui_ux_designer': {
                'exclude_skills': ['python', 'java', 'sql', 'backend_development'],
                'exclude_categories': ['programming_language', 'database']
            },
            'digital_marketing_specialist': {
                'exclude_skills': ['python', 'java', 'docker', 'kubernetes'],
                'exclude_categories': ['programming_language', 'cloud_service']
            }
        }
        
        logger.info(f"✅ 加载了 {len(self.explicit_role_mappings)} 个角色和 {len(self.explicit_industry_mappings)} 个行业的显式映射")
    
    def _load_semantic_descriptions(self):
        """加载语义描述 - 优化版本，减少重叠"""
        # 更精确的角色语义描述，减少重叠
        self.role_semantic_descriptions = {
            'software_engineer': 'general software development programming coding applications systems architecture',
            'data_scientist': 'data analysis statistics machine learning predictive modeling analytics research insights',
            'product_manager': 'product strategy roadmap requirements business analysis market research user needs',
            'digital_marketing_specialist': 'digital marketing campaigns advertising social media brand promotion customer acquisition',
            'mechanical_engineer': 'mechanical design manufacturing CAD materials physics mechanics systems',
            'electrical_engineer': 'electrical circuits electronics power systems embedded hardware automation control',
            'automotive_engineer': 'vehicle automotive transportation mobility electric autonomous powertrain safety',
            'business_analyst': 'business process analysis requirements gathering stakeholder consultation improvement optimization',
            'project_manager': 'project coordination planning scheduling resource management delivery execution',
            'cybersecurity_specialist': 'security protection threat vulnerability risk assessment compliance audit',
            'devops_engineer': 'deployment automation infrastructure orchestration continuous integration delivery operations',
            'cloud_engineer': 'cloud infrastructure scalability distributed systems serverless containerization',
            'ai_ml_engineer': 'artificial intelligence machine learning deep learning neural networks algorithms',
            'frontend_web_developer': 'user interface web frontend javascript html css responsive interactive',
            'backend_developer': 'server backend api database microservices architecture scalability performance',
            'mobile_app_developer': 'mobile applications ios android native cross-platform mobile user experience',
            'ui_ux_designer': 'user interface user experience design visual interaction usability wireframes prototypes',
            'quality_assurance_engineer': 'testing quality assurance automation bug verification validation',
            'systems_administrator': 'systems administration server infrastructure maintenance monitoring troubleshooting',
            'database_administrator': 'database administration performance optimization backup recovery maintenance'
        }
        
        # 更精确的行业语义描述
        self.industry_semantic_descriptions = {
            'consulting': 'consulting advisory strategy transformation change management business solutions',
            'finance': 'financial banking investment trading wealth management risk assessment regulatory',
            'tech': 'technology software digital innovation programming computer systems information',
            'law': 'legal litigation corporate law intellectual property compliance regulatory affairs',
            'automotive': 'automotive vehicles transportation manufacturing electric autonomous mobility',
            'healthcare': 'healthcare medical clinical pharmaceutical biotechnology patient care devices',
            'media': 'media entertainment content creation broadcasting journalism creative arts',
            'real_estate': 'real estate property development construction architecture management residential commercial',
            'energy': 'energy renewable power sustainability oil gas solar wind nuclear infrastructure',
            'manufacturing': 'manufacturing production industrial operations quality control supply chain',
            'retail': 'retail ecommerce sales customer service merchandising consumer goods',
            'education': 'education learning training academic teaching development research',
            'aerospace': 'aerospace aviation flight space aircraft defense satellite systems',
            'chemicals': 'chemical manufacturing materials petrochemical pharmaceutical industrial chemistry',
            'telecommunications': 'telecommunications networking communication wireless connectivity infrastructure',
            'food': 'food beverage restaurant culinary agriculture nutrition processing hospitality',
            'logistics': 'logistics transportation supply chain shipping warehouse distribution',
            'gaming': 'gaming entertainment interactive esports game development virtual reality',
            'fashion': 'fashion design textile apparel luxury retail brand creative',
            'sports': 'sports fitness athletic recreation wellness health management',
            'agriculture': 'agriculture farming crops livestock sustainable food production',
            'mining': 'mining extraction minerals resources geological engineering',
            'hospitality': 'hospitality hotel travel tourism leisure customer service',
            'security': 'security protection safety risk management surveillance',
            'entertainment': 'entertainment media content film music television arts',
            'government': 'government public policy administration civic services regulatory',
            'nonprofit': 'nonprofit charity social philanthropy community development',
            'biotechnology': 'biotechnology life sciences genetics molecular biology research',
            'insurance': 'insurance risk assessment actuarial underwriting claims protection',
            'architecture': 'architecture design building construction planning urban structural',
            'environmental': 'environmental sustainability green technology climate ecology conservation'
        }
    
    def classify_skills(self, all_skills: List[Dict]) -> Tuple[Dict, Dict]:
        """分级分类所有技能"""
        logger.info("🧠 开始分级智能技能分类...")
        
        # 初始化分类结构
        role_classifications = {role: {
            'essential': [], 'common': [], 'collaboration': [], 'specializations': []
        } for role in self.role_semantic_descriptions.keys()}
        
        industry_classifications = {industry: {
            'core_domain': [], 'regulatory': [], 'business_focus': [], 'unique_requirements': []
        } for industry in self.industry_semantic_descriptions.keys()}
        
        # 预计算embeddings
        self._precompute_embeddings()
        
        # 统计变量
        stats = {
            'explicit_role_assigned': 0,
            'explicit_industry_assigned': 0,
            'semantic_role_assigned': 0,
            'semantic_industry_assigned': 0,
            'unassigned': 0,
            'skipped_by_exclusion': 0
        }
        
        # 收集已分配的技能
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
        for skill_info in all_skills:
            skill = skill_info['skill']
            
            if skill in assigned_skills:
                continue
                
            # 检查排除规则
            if self._is_excluded_by_rules(skill_info):
                stats['skipped_by_exclusion'] += 1
                continue
            
            # 高置信度角色匹配
            best_role, role_similarity = self._calculate_best_role_match(skill_info)
            if best_role and role_similarity >= self.high_confidence_threshold:
                category = self._determine_role_category(skill_info, role_similarity)
                role_classifications[best_role][category].append(skill)
                assigned_skills.add(skill)
                stats['semantic_role_assigned'] += 1
                continue
            
            # 高置信度行业匹配
            best_industry, industry_similarity = self._calculate_best_industry_match(skill_info)
            if best_industry and industry_similarity >= self.high_confidence_threshold:
                category = self._determine_industry_category(skill_info, industry_similarity)
                industry_classifications[best_industry][category].append(skill)
                assigned_skills.add(skill)
                stats['semantic_industry_assigned'] += 1
        
        # === 第三轮：中等置信度语义匹配 ===
        logger.info("🔍 第三轮：中等置信度语义匹配...")
        for skill_info in all_skills:
            skill = skill_info['skill']
            
            if skill in assigned_skills:
                continue
                
            if self._is_excluded_by_rules(skill_info):
                continue
            
            # 中等置信度角色匹配
            best_role, role_similarity = self._calculate_best_role_match(skill_info)
            if best_role and role_similarity >= self.medium_confidence_threshold:
                category = self._determine_role_category(skill_info, role_similarity)
                role_classifications[best_role][category].append(skill)
                assigned_skills.add(skill)
                stats['semantic_role_assigned'] += 1
                continue
            
            # 中等置信度行业匹配
            best_industry, industry_similarity = self._calculate_best_industry_match(skill_info)
            if best_industry and industry_similarity >= self.medium_confidence_threshold:
                category = self._determine_industry_category(skill_info, industry_similarity)
                industry_classifications[best_industry][category].append(skill)
                assigned_skills.add(skill)
                stats['semantic_industry_assigned'] += 1
        
        # === 第四轮：兜底处理 ===
        logger.info("🥅 第四轮：兜底处理...")
        for skill_info in all_skills:
            skill = skill_info['skill']
            
            if skill in assigned_skills:
                continue
            
            # 尝试低置信度匹配
            best_role, role_similarity = self._calculate_best_role_match(skill_info)
            best_industry, industry_similarity = self._calculate_best_industry_match(skill_info)
            
            # 选择更好的匹配
            if role_similarity >= self.low_confidence_threshold and role_similarity > industry_similarity:
                category = self._determine_role_category(skill_info, role_similarity)
                role_classifications[best_role][category].append(skill)
                stats['semantic_role_assigned'] += 1
            elif industry_similarity >= self.low_confidence_threshold:
                category = self._determine_industry_category(skill_info, industry_similarity)
                industry_classifications[best_industry][category].append(skill)
                stats['semantic_industry_assigned'] += 1
            else:
                # 最终兜底到tech行业
                industry_classifications['tech']['core_domain'].append(skill)
                stats['unassigned'] += 1
        
        # 清理空分类
        role_classifications = self._clean_classifications(role_classifications)
        industry_classifications = self._clean_classifications(industry_classifications)
        
        # 输出统计信息
        total_processed = sum(stats.values())
        logger.info(f"📊 分级分类完成:")
        logger.info(f"  📋 显式角色分配: {stats['explicit_role_assigned']}")
        logger.info(f"  🏢 显式行业分配: {stats['explicit_industry_assigned']}")
        logger.info(f"  🎯 语义角色分配: {stats['semantic_role_assigned']}")
        logger.info(f"  🏭 语义行业分配: {stats['semantic_industry_assigned']}")
        logger.info(f"  🥅 兜底分配: {stats['unassigned']}")
        logger.info(f"  ❌ 排除跳过: {stats['skipped_by_exclusion']}")
        logger.info(f"  📈 总处理: {len(all_skills)} 个技能")
        
        return role_classifications, industry_classifications
    
    def _precompute_embeddings(self):
        """预计算embeddings"""
        logger.info("🧬 预计算embedding向量...")
        
        self.role_embeddings = {}
        for role, description in self.role_semantic_descriptions.items():
            self.role_embeddings[role] = self.model.encode([description])[0]
        
        self.industry_embeddings = {}
        for industry, description in self.industry_semantic_descriptions.items():
            self.industry_embeddings[industry] = self.model.encode([description])[0]
        
        logger.info(f"✅ 预计算完成: {len(self.role_embeddings)} 角色, {len(self.industry_embeddings)} 行业")
    
    def _calculate_best_role_match(self, skill_info: Dict) -> Tuple[Optional[str], float]:
        """计算最佳角色匹配"""
        enhanced_skill = self.enhance_skill_description(skill_info)
        
        try:
            skill_embedding = self.model.encode([enhanced_skill])[0]
        except Exception:
            return None, 0.0
        
        best_role, best_similarity = None, 0.0
        
        for role, role_embedding in self.role_embeddings.items():
            similarity = cosine_similarity([skill_embedding], [role_embedding])[0][0]
            bonus = self._calculate_role_type_bonus(skill_info, role)
            adjusted_similarity = similarity * (1 + bonus)
            
            if adjusted_similarity > best_similarity:
                best_similarity = adjusted_similarity
                best_role = role
        
        return best_role, best_similarity
    
    def _calculate_best_industry_match(self, skill_info: Dict) -> Tuple[Optional[str], float]:
        """计算最佳行业匹配"""
        enhanced_skill = self.enhance_skill_description(skill_info)
        
        try:
            skill_embedding = self.model.encode([enhanced_skill])[0]
        except Exception:
            return None, 0.0
        
        best_industry, best_similarity = None, 0.0
        
        for industry, industry_embedding in self.industry_embeddings.items():
            similarity = cosine_similarity([skill_embedding], [industry_embedding])[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_industry = industry
        
        return best_industry, best_similarity
    
    def _calculate_role_type_bonus(self, skill_info: Dict, role: str) -> float:
        """计算角色类型加分 - 降低加分幅度"""
        skill_type = skill_info['skill_type']
        skill = skill_info['skill'].lower()
        
        # 减少加分幅度，避免过度偏向某些角色
        role_type_affinity = {
            'software_engineer': ['programming_language', 'framework'],
            'data_scientist': ['programming_language', 'tool'],
            'devops_engineer': ['cloud_service', 'tool'],
            'cloud_engineer': ['cloud_service'],
            'frontend_web_developer': ['programming_language', 'framework'],
            'backend_developer': ['programming_language', 'database'],
            'ui_ux_designer': ['tool'],
            'project_manager': ['methodology'],
            'cybersecurity_specialist': ['certification', 'tool']
        }
        
        # 降低加分比例从0.2降到0.1 (10%)
        if role in role_type_affinity and skill_type in role_type_affinity[role]:
            return 0.1
        
        # 特殊技能直接匹配 - 也降低加分
        special_matches = {
            'figma': ['ui_ux_designer'],
            'sketch': ['ui_ux_designer'],
            'adobe_xd': ['ui_ux_designer'],
            'google_ads': ['digital_marketing_specialist'],
            'facebook_ads': ['digital_marketing_specialist'],
            'jira': ['project_manager'],
            'scrum': ['project_manager'],
            'agile': ['project_manager'],
            'tableau': ['data_scientist'],
            'kubernetes': ['devops_engineer', 'cloud_engineer'],
            'docker': ['devops_engineer', 'cloud_engineer']
        }
        
        # 降低特殊匹配加分从0.3降到0.15 (15%)
        if skill in special_matches and role in special_matches[skill]:
            return 0.15
        
        return 0.0
    
    def _determine_role_category(self, skill_info: Dict, similarity: float) -> str:
        """确定角色类别 - 提高阈值"""
        skill_type = skill_info['skill_type']
        
        # 提高各个阈值，使分类更加严格
        if similarity >= 0.85:  # 从0.8提高到0.85
            return 'essential'
        elif similarity >= 0.75:  # 从0.7提高到0.75
            return 'essential' if skill_type in ['programming_language', 'framework', 'database'] else 'common'
        elif similarity >= 0.65:  # 从0.6提高到0.65
            return 'specializations' if skill_type in ['certification', 'methodology'] else 'common'
        else:
            return 'collaboration'
    
    def _determine_industry_category(self, skill_info: Dict, similarity: float) -> str:
        """确定行业类别"""
        skill_type = skill_info['skill_type']
        skill = skill_info['skill'].lower()
        
        if skill_type == 'certification' or skill_type == 'language':
            return 'unique_requirements'
        elif 'compliance' in skill or 'regulatory' in skill or 'gdpr' in skill or 'iso' in skill:
            return 'regulatory'
        elif skill_type == 'methodology' or 'management' in skill or 'strategy' in skill:
            return 'business_focus'
        else:
            return 'core_domain'
    
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
            vector_data = self._generate_vectors_for_role(categories)
            
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
            vector_data = self._generate_vectors_for_industry(categories)
            
            industry_baselines[industry] = {
                'semantic_data': semantic_data,
                'vector_data': vector_data
            }
        
        return industry_baselines
    
    def _generate_vectors_for_skills(self, skills: List[str]) -> Dict:
        """为技能列表生成向量数据"""
        if not skills:
            return {}
        
        try:
            # 使用向量生成模型计算embeddings
            skill_embeddings = self.vector_model.encode(skills)
            
            # 计算全局向量（平均向量）
            global_vector = np.mean(skill_embeddings, axis=0) if len(skill_embeddings) > 0 else np.zeros(384)
            
            return {
                'global_vector': global_vector.tolist(),
                'skill_vectors': {skill: embedding.tolist() for skill, embedding in zip(skills, skill_embeddings)},
                'vector_metadata': {
                    'total_vectors': len(skills),
                    'vector_dimension': 384,
                    'model_used': 'all-MiniLM-L6-v2'
                }
            }
        except Exception as e:
            logger.error(f"向量生成失败: {e}")
            return {}
    
    def _generate_vectors_for_role(self, categories: Dict) -> Dict:
        """为角色分类生成向量数据"""
        all_skills = []
        for skills in categories.values():
            all_skills.extend(skills)
        
        if not all_skills:
            return {}
        
        try:
            # 生成所有技能的向量
            unique_skills = list(set(all_skills))
            skill_embeddings = self.vector_model.encode(unique_skills)
            skill_vectors = {skill: embedding.tolist() for skill, embedding in zip(unique_skills, skill_embeddings)}
            
            # 计算各类别的平均向量
            category_vectors = {}
            for category, skills in categories.items():
                if skills:
                    category_embeddings = [self.vector_model.encode([skill])[0] for skill in skills]
                    if category_embeddings:
                        category_vectors[category] = np.mean(category_embeddings, axis=0).tolist()
            
            # 计算整体角色向量
            role_vector = np.mean(skill_embeddings, axis=0) if len(skill_embeddings) > 0 else np.zeros(384)
            
            return {
                'role_vector': role_vector.tolist(),
                'category_vectors': category_vectors,
                'skill_vectors': skill_vectors,
                'vector_metadata': {
                    'total_vectors': len(unique_skills),
                    'vector_dimension': 384,
                    'model_used': 'all-MiniLM-L6-v2'
                }
            }
        except Exception as e:
            logger.error(f"角色向量生成失败: {e}")
            return {}
    
    def _generate_vectors_for_industry(self, categories: Dict) -> Dict:
        """为行业分类生成向量数据"""
        all_skills = []
        for skills in categories.values():
            all_skills.extend(skills)
        
        if not all_skills:
            return {}
        
        try:
            # 生成所有技能的向量
            unique_skills = list(set(all_skills))
            skill_embeddings = self.vector_model.encode(unique_skills)
            skill_vectors = {skill: embedding.tolist() for skill, embedding in zip(unique_skills, skill_embeddings)}
            
            # 计算各类别的平均向量
            category_vectors = {}
            for category, skills in categories.items():
                if skills:
                    category_embeddings = [self.vector_model.encode([skill])[0] for skill in skills]
                    if category_embeddings:
                        category_vectors[category] = np.mean(category_embeddings, axis=0).tolist()
            
            # 计算整体行业向量
            industry_vector = np.mean(skill_embeddings, axis=0) if len(skill_embeddings) > 0 else np.zeros(384)
            
            return {
                'industry_vector': industry_vector.tolist(),
                'category_vectors': category_vectors,
                'skill_vectors': skill_vectors,
                'vector_metadata': {
                    'total_vectors': len(unique_skills),
                    'vector_dimension': 384,
                    'model_used': 'all-MiniLM-L6-v2'
                }
            }
        except Exception as e:
            logger.error(f"行业向量生成失败: {e}")
            return {}

class DatabaseManager:
    """数据库操作管理器 - 分离语义数据和向量数据"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.supabase = self._init_supabase()
    
    def _init_supabase(self) -> Optional[Client]:
        """初始化Supabase客户端"""
        if not self.config.supabase_url or not self.config.supabase_key:
            logger.warning("⚠️ Supabase配置缺失，跳过数据库操作")
            return None
        
        try:
            client = create_client(self.config.supabase_url, self.config.supabase_key)
            logger.info("✅ Supabase客户端初始化成功")
            return client
        except Exception as e:
            logger.error(f"❌ Supabase客户端初始化失败: {e}")
            return None
    
    def upload_baselines(self, baselines: Dict) -> Dict:
        """上传基线数据到数据库（分离存储）"""
        if not self.supabase:
            logger.warning("⚠️ 数据库未连接，跳过上传")
            return {'status': 'skipped', 'reason': 'no_database_connection'}
        
        logger.info("📤 开始上传基线数据到分离的表...")
        results = {}
        
        # 上传全局基线
        results['global'] = self._upload_baseline_pair(
            'global', 'global_baseline', baselines['global']
        )
        
        # 上传角色基线
        results['roles'] = {}
        for role, baseline_data in baselines['roles'].items():
            results['roles'][role] = self._upload_baseline_pair(
                'role', f'{role}_baseline', baseline_data
            )
        
        # 上传行业基线
        results['industries'] = {}
        for industry, baseline_data in baselines['industries'].items():
            results['industries'][industry] = self._upload_baseline_pair(
                'industry', f'{industry}_baseline', baseline_data
            )
        
        logger.info("✅ 基线数据上传完成")
        return results
    
    def _upload_baseline_pair(self, baseline_type: str, name: str, baseline_data: Dict) -> Dict:
        """上传基线数据对（语义数据 + 向量数据）"""
        try:
            semantic_data = baseline_data.get('semantic_data', {})
            vector_data = baseline_data.get('vector_data', {})
            
            if not semantic_data:
                logger.warning(f"基线 {name} 缺少语义数据，跳过上传")
                return {'status': 'error', 'error': 'missing_semantic_data'}
            
            # 1. 首先上传语义数据到 semantic_baselines 表
            semantic_result = self._upload_semantic_baseline(baseline_type, name, semantic_data)
            
            if semantic_result['status'] != 'uploaded':
                logger.error(f"语义数据上传失败，跳过向量数据上传: {name}")
                return semantic_result
            
            baseline_id = semantic_result['id']
            
            # 2. 然后上传向量数据到 baseline_vectors 表
            if vector_data:
                vector_result = self._upload_vector_data(baseline_id, vector_data)
                
                return {
                    'status': 'uploaded',
                    'baseline_id': baseline_id,
                    'semantic_upload': semantic_result,
                    'vector_upload': vector_result
                }
            else:
                logger.warning(f"基线 {name} 缺少向量数据")
                return {
                    'status': 'partial',
                    'baseline_id': baseline_id,
                    'semantic_upload': semantic_result,
                    'vector_upload': {'status': 'skipped', 'reason': 'no_vector_data'}
                }
                
        except Exception as e:
            logger.error(f"上传基线对 {name} 失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _upload_semantic_baseline(self, baseline_type: str, name: str, semantic_data: Dict) -> Dict:
        """上传语义数据到 semantic_baselines 表"""
        source_hash = self._calculate_hash(semantic_data)
        
        # 检查是否已存在相同的语义数据
        try:
            result = self.supabase.table("semantic_baselines") \
                .select("*") \
                .eq("source_hash", source_hash) \
                .execute()
            
            if result.data:
                logger.info(f"语义基线 {name} 已存在，跳过上传")
                return {'status': 'exists', 'id': result.data[0]['id']}
        except Exception as e:
            logger.error(f"检查语义基线 {name} 时出错: {e}")
            return {'status': 'error', 'error': str(e)}
        
        # 插入新的语义数据
        insert_data = {
            "baseline_type": baseline_type,
            "name": name,
            "version": 1,
            "is_active": True,
            "baseline_data": semantic_data,  # 只包含人类可读的数据，不包含向量
            "source_hash": source_hash
        }
        
        try:
            result = self.supabase.table("semantic_baselines") \
                .insert(insert_data) \
                .execute()
            
            logger.info(f"✅ 成功上传语义基线: {name}")
            return {'status': 'uploaded', 'id': result.data[0]['id']}
        except Exception as e:
            logger.error(f"❌ 上传语义基线 {name} 失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _upload_vector_data(self, baseline_id: int, vector_data: Dict) -> Dict:
        """上传向量数据到 baseline_vectors 表"""
        # 检查是否已存在该baseline_id的向量数据
        try:
            result = self.supabase.table("baseline_vectors") \
                .select("*") \
                .eq("baseline_id", baseline_id) \
                .eq("embedding_model", "all-MiniLM-L6-v2") \
                .execute()
            
            if result.data:
                logger.info(f"向量数据 baseline_id={baseline_id} 已存在，跳过上传")
                return {'status': 'exists', 'id': result.data[0]['id']}
        except Exception as e:
            logger.error(f"检查向量数据 baseline_id={baseline_id} 时出错: {e}")
            return {'status': 'error', 'error': str(e)}
        
        # 插入新的向量数据
        insert_data = {
            "baseline_id": baseline_id,
            "embedding_model": "all-MiniLM-L6-v2",
            "vectors_data": vector_data  # 包含所有向量相关数据
        }
        
        try:
            result = self.supabase.table("baseline_vectors") \
                .insert(insert_data) \
                .execute()
            
            logger.info(f"✅ 成功上传向量数据: baseline_id={baseline_id}")
            return {'status': 'uploaded', 'id': result.data[0]['id']}
        except Exception as e:
            logger.error(f"❌ 上传向量数据 baseline_id={baseline_id} 失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_hash(self, data: dict) -> str:
        """计算数据哈希值"""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

class PipelineOrchestrator:
    """整体流程编排器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.processor = SkillProcessor()
        self.classifier = SemanticClassifier(config)
        # 修改初始化基线生成器的调用
        self.baseline_generator = BaselineGenerator(config, self.classifier.model)
        self.database_manager = DatabaseManager(config)
        
        # 创建输出目录
        Path(self.config.output_dir).mkdir(exist_ok=True)
    
    def run_complete_pipeline(self, vocab_path: str) -> Dict:
        """运行完整管线"""
        logger.info("🚀 启动端到端技能处理管线")
        logger.info(f"📁 输入文件: {vocab_path}")
        logger.info(f"📁 输出目录: {self.config.output_dir}")
        print("-" * 60)
        
        pipeline_start = datetime.now()
        results = {}
        
        try:
            # 1. 加载和处理词汇表
            logger.info("📊 Step 1: 加载技能词汇表...")
            skills_vocab = self._load_vocabulary(vocab_path)
            all_skills = self.processor.extract_all_skills(skills_vocab)
            logger.info(f"📋 提取到 {len(all_skills)} 个技能")
            results['skills_extracted'] = len(all_skills)
            
            # 2. 语义分类
            logger.info("🧠 Step 2: 执行语义分类...")
            role_classifications, industry_classifications = self.classifier.classify_skills(all_skills)
            results['role_classifications'] = role_classifications
            results['industry_classifications'] = industry_classifications
            
            # 保存分类结果
            self._save_classifications(role_classifications, industry_classifications)
            
            # 3. 生成语义基线
            logger.info("🏗️ Step 3: 生成语义基线...")
            baselines = self.baseline_generator.generate_all_baselines(
                role_classifications, industry_classifications
            )
            results['baselines'] = baselines
            
            # 保存基线文件
            self._save_baselines(baselines)
            
            # 4. 上传到数据库
            logger.info("📤 Step 4: 上传到数据库...")
            upload_results = self.database_manager.upload_baselines(baselines)
            results['upload_results'] = upload_results
            
            # 5. 生成总结报告
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()
            
            results['summary'] = self._generate_summary_report(
                all_skills, role_classifications, industry_classifications, 
                baselines, upload_results, duration
            )
            
            # 保存总结报告
            self._save_summary_report(results['summary'])
            
            logger.info("🎉 管线执行完成!")
            self._print_summary(results['summary'])
            
        except Exception as e:
            logger.error(f"❌ 管线执行失败: {e}")
            results['error'] = str(e)
            raise
        
        return results
    
    def _load_vocabulary(self, vocab_path: str) -> Dict:
        """加载技能词汇表"""
        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(f"词汇表文件不存在: {vocab_path}")
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 尝试直接解析JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 如果失败，尝试从markdown中提取JSON
            import re
            json_match = re.search(r'\{[\s\S]*?\}(?=\s*###|\s*$)', content)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("无法解析技能词汇表文件")
    
    def _save_classifications(self, role_classifications: Dict, industry_classifications: Dict):
        """保存分类结果"""
        # 保存角色分类
        role_path = Path(self.config.output_dir) / "role_skill_classifications.json"
        with open(role_path, 'w', encoding='utf-8') as f:
            json.dump(role_classifications, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 角色分类已保存: {role_path}")
        
        # 保存行业分类
        industry_path = Path(self.config.output_dir) / "industry_skill_classifications.json"
        with open(industry_path, 'w', encoding='utf-8') as f:
            json.dump(industry_classifications, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 行业分类已保存: {industry_path}")
        
        # 保存分类详情（如果启用调试）
        if self.config.save_classification_details:
            self._save_classification_details(role_classifications, industry_classifications)
    
    def _save_classification_details(self, role_classifications: Dict, industry_classifications: Dict):
        """保存分类详情统计"""
        details = {
            'classification_summary': {
                'total_roles_with_skills': len([r for r in role_classifications.values() if any(skills for skills in r.values())]),
                'total_industries_with_skills': len([i for i in industry_classifications.values() if any(skills for skills in i.values())]),
                'role_skill_distribution': {
                    role: {cat: len(skills) for cat, skills in categories.items() if skills}
                    for role, categories in role_classifications.items()
                    if any(skills for skills in categories.values())
                },
                'industry_skill_distribution': {
                    industry: {cat: len(skills) for cat, skills in categories.items() if skills}
                    for industry, categories in industry_classifications.items()
                    if any(skills for skills in categories.values())
                }
            },
            'quality_insights': {
                'largest_role_categories': self._get_largest_categories(role_classifications),
                'largest_industry_categories': self._get_largest_categories(industry_classifications),
                'potential_issues': self._identify_potential_issues(role_classifications, industry_classifications)
            },
            'configuration_used': {
                'high_confidence_threshold': self.config.high_confidence_threshold,
                'medium_confidence_threshold': self.config.medium_confidence_threshold,
                'low_confidence_threshold': self.config.low_confidence_threshold,
                'model_name': self.config.classification_model
            }
        }
        
        details_path = Path(self.config.output_dir) / "classification_details.json"
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        logger.info(f"📋 分类详情已保存: {details_path}")
    
    def _get_largest_categories(self, classifications: Dict) -> List[Dict]:
        """获取最大的分类类别"""
        largest = []
        for name, categories in classifications.items():
            for category, skills in categories.items():
                if skills:
                    largest.append({
                        'name': name,
                        'category': category,
                        'skill_count': len(skills),
                        'sample_skills': skills[:5]  # 只保存前5个技能作为示例
                    })
        
        # 按技能数量排序
        largest.sort(key=lambda x: x['skill_count'], reverse=True)
        return largest[:10]  # 返回前10个最大的类别
    
    def _identify_potential_issues(self, role_classifications: Dict, industry_classifications: Dict) -> List[str]:
        """识别潜在的分类问题"""
        issues = []
        
        # 检查是否有角色占用过多技能
        total_role_skills = sum(sum(len(skills) for skills in categories.values()) 
                              for categories in role_classifications.values())
        
        for role, categories in role_classifications.items():
            role_skills = sum(len(skills) for skills in categories.values())
            if role_skills > total_role_skills * 0.3:  # 如果单个角色占用超过30%的技能
                issues.append(f"角色 '{role}' 可能占用了过多技能 ({role_skills} 个, {role_skills/total_role_skills*100:.1f}%)")
        
        # 检查是否有行业为空
        empty_industries = [industry for industry, categories in industry_classifications.items() 
                          if not any(skills for skills in categories.values())]
        if len(empty_industries) > len(industry_classifications) * 0.5:
            issues.append(f"超过一半的行业分类为空 ({len(empty_industries)}/{len(industry_classifications)})")
        
        # 检查特定技能是否被正确分类
        design_tools = ['figma', 'sketch', 'adobe_xd']
        ui_ux_skills = []
        if 'ui_ux_designer' in role_classifications:
            ui_ux_skills = [skill for category in role_classifications['ui_ux_designer'].values() 
                           for skill in category]
        
        missing_design_tools = [tool for tool in design_tools if tool not in ui_ux_skills]
        if missing_design_tools:
            issues.append(f"设计工具可能未正确分类到UI/UX设计师: {missing_design_tools}")
        
        return issues
    
    def _save_baselines(self, baselines: Dict):
        """保存基线文件（分离语义和向量数据）"""
        # 保存全局基线 - 语义数据
        global_semantic_path = Path(self.config.output_dir) / "global_baseline_semantic.json"
        with open(global_semantic_path, 'w', encoding='utf-8') as f:
            json.dump(baselines['global']['semantic_data'], f, indent=2, ensure_ascii=False)
        logger.info(f"💾 全局语义基线已保存: {global_semantic_path}")
        
        # 保存全局基线 - 向量数据
        global_vector_path = Path(self.config.output_dir) / "global_baseline_vectors.json"
        with open(global_vector_path, 'w', encoding='utf-8') as f:
            json.dump(baselines['global']['vector_data'], f, indent=2, ensure_ascii=False)
        logger.info(f"💾 全局向量基线已保存: {global_vector_path}")
        
        # 保存角色基线 - 语义数据
        role_semantic_data = {role: data['semantic_data'] for role, data in baselines['roles'].items()}
        role_semantic_path = Path(self.config.output_dir) / "role_baselines_semantic.json"
        with open(role_semantic_path, 'w', encoding='utf-8') as f:
            json.dump(role_semantic_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 角色语义基线已保存: {role_semantic_path}")
        
        # 保存角色基线 - 向量数据
        role_vector_data = {role: data['vector_data'] for role, data in baselines['roles'].items()}
        role_vector_path = Path(self.config.output_dir) / "role_baselines_vectors.json"
        with open(role_vector_path, 'w', encoding='utf-8') as f:
            json.dump(role_vector_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 角色向量基线已保存: {role_vector_path}")
        
        # 保存行业基线 - 语义数据
        industry_semantic_data = {industry: data['semantic_data'] for industry, data in baselines['industries'].items()}
        industry_semantic_path = Path(self.config.output_dir) / "industry_baselines_semantic.json"
        with open(industry_semantic_path, 'w', encoding='utf-8') as f:
            json.dump(industry_semantic_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 行业语义基线已保存: {industry_semantic_path}")
        
        # 保存行业基线 - 向量数据
        industry_vector_data = {industry: data['vector_data'] for industry, data in baselines['industries'].items()}
        industry_vector_path = Path(self.config.output_dir) / "industry_baselines_vectors.json"
        with open(industry_vector_path, 'w', encoding='utf-8') as f:
            json.dump(industry_vector_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 行业向量基线已保存: {industry_vector_path}")
        
    def _count_semantic_uploads(self, upload_results: Dict) -> int:
        """统计语义数据上传数量"""
        count = 0
        if upload_results.get('global', {}).get('semantic_upload', {}).get('status') in ['uploaded', 'exists']:
            count += 1
        for result in upload_results.get('roles', {}).values():
            if result.get('semantic_upload', {}).get('status') in ['uploaded', 'exists']:
                count += 1
        for result in upload_results.get('industries', {}).values():
            if result.get('semantic_upload', {}).get('status') in ['uploaded', 'exists']:
                count += 1
        return count

    def _count_vector_uploads(self, upload_results: Dict) -> int:
        """统计向量数据上传数量"""
        count = 0
        if upload_results.get('global', {}).get('vector_upload', {}).get('status') in ['uploaded', 'exists']:
            count += 1
        for result in upload_results.get('roles', {}).values():
            if result.get('vector_upload', {}).get('status') in ['uploaded', 'exists']:
                count += 1
        for result in upload_results.get('industries', {}).values():
            if result.get('vector_upload', {}).get('status') in ['uploaded', 'exists']:
                count += 1
        return count

    def _generate_summary_report(self, all_skills: List[Dict], role_classifications: Dict, 
                                industry_classifications: Dict, baselines: Dict, 
                                upload_results: Dict, duration: float) -> Dict:
        """生成总结报告"""
        # 统计分类结果
        total_role_skills = sum(sum(len(skills) for skills in categories.values()) 
                              for categories in role_classifications.values())
        total_industry_skills = sum(sum(len(skills) for skills in categories.values()) 
                                  for categories in industry_classifications.values())
        
        # 统计上传结果 - 更新以支持新的分离上传格式
        upload_stats = {
            'global_uploaded': upload_results.get('global', {}).get('status') == 'uploaded',
            'roles_uploaded': sum(1 for result in upload_results.get('roles', {}).values() 
                                if result.get('status') == 'uploaded'),
            'industries_uploaded': sum(1 for result in upload_results.get('industries', {}).values() 
                                     if result.get('status') == 'uploaded'),
            'total_semantic_uploads': self._count_semantic_uploads(upload_results),
            'total_vector_uploads': self._count_vector_uploads(upload_results),
            'total_errors': sum(1 for section in upload_results.values() 
                              for result in (section.values() if isinstance(section, dict) else [section])
                              if result.get('status') == 'error')
        }
        
        return {
            'pipeline_info': {
                'execution_time_seconds': round(duration, 2),
                'completion_time': datetime.now().isoformat(),
                'config': {
                    'model_name': self.config.classification_model,
                    'role_threshold': self.config.role_threshold,
                    'industry_threshold': self.config.industry_threshold,
                    'min_skills_for_baseline': self.config.min_skills_for_baseline
                }
            },
            'skill_processing': {
                'total_skills_extracted': len(all_skills),
                'total_role_assignments': total_role_skills,
                'total_industry_assignments': total_industry_skills,
                'roles_with_skills': len(role_classifications),
                'industries_with_skills': len(industry_classifications)
            },
            'baseline_generation': {
                'global_baseline_generated': 'global' in baselines,
                'role_baselines_generated': len(baselines.get('roles', {})),
                'industry_baselines_generated': len(baselines.get('industries', {})),
                'total_baselines': 1 + len(baselines.get('roles', {})) + len(baselines.get('industries', {}))
            },
            'database_upload': upload_stats,
            'quality_metrics': {
                'classification_coverage': {
                    'role_coverage_percent': round((total_role_skills / len(all_skills)) * 100, 2) if all_skills else 0,
                    'industry_coverage_percent': round((total_industry_skills / len(all_skills)) * 100, 2) if all_skills else 0
                },
                'baseline_quality': {
                    'avg_skills_per_role': round(total_role_skills / len(role_classifications), 2) if role_classifications else 0,
                    'avg_skills_per_industry': round(total_industry_skills / len(industry_classifications), 2) if industry_classifications else 0
                }
            },
            'output_files': {
                'role_classifications': f"{self.config.output_dir}/role_skill_classifications.json",
                'industry_classifications': f"{self.config.output_dir}/industry_skill_classifications.json",
                # 新的分离格式文件
                'global_baseline_semantic': f"{self.config.output_dir}/global_baseline_semantic.json",
                'global_baseline_vectors': f"{self.config.output_dir}/global_baseline_vectors.json",
                'role_baselines_semantic': f"{self.config.output_dir}/role_baselines_semantic.json",
                'role_baselines_vectors': f"{self.config.output_dir}/role_baselines_vectors.json",
                'industry_baselines_semantic': f"{self.config.output_dir}/industry_baselines_semantic.json",
                'industry_baselines_vectors': f"{self.config.output_dir}/industry_baselines_vectors.json",
                'summary_report': f"{self.config.output_dir}/pipeline_summary.json"
            }
        }
    
    def _save_summary_report(self, summary: Dict):
        """保存总结报告"""
        summary_path = Path(self.config.output_dir) / "pipeline_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 总结报告已保存: {summary_path}")
    
    def _print_summary(self, summary: Dict):
        """打印总结信息"""
        print("\n" + "="*60)
        print("📊 管线执行总结")
        print("="*60)
        
        # 基本信息
        pipeline_info = summary['pipeline_info']
        print(f"⏱️  执行时间: {pipeline_info['execution_time_seconds']} 秒")
        print(f"🕐 完成时间: {pipeline_info['completion_time']}")
        
        # 技能处理
        skill_info = summary['skill_processing']
        print(f"\n📋 技能处理:")
        print(f"   总技能数: {skill_info['total_skills_extracted']}")
        print(f"   角色分配: {skill_info['total_role_assignments']}")
        print(f"   行业分配: {skill_info['total_industry_assignments']}")
        print(f"   有效角色: {skill_info['roles_with_skills']}")
        print(f"   有效行业: {skill_info['industries_with_skills']}")
        
        # 基线生成
        baseline_info = summary['baseline_generation']
        print(f"\n🏗️ 基线生成:")
        print(f"   总基线数: {baseline_info['total_baselines']}")
        print(f"   角色基线: {baseline_info['role_baselines_generated']}")
        print(f"   行业基线: {baseline_info['industry_baselines_generated']}")
        
        # 数据库上传
        upload_info = summary['database_upload']
        print(f"\n📤 数据库上传:")
        print(f"   全局基线: {'✅' if upload_info['global_uploaded'] else '❌'}")
        print(f"   角色基线: {upload_info['roles_uploaded']} 个")
        print(f"   行业基线: {upload_info['industries_uploaded']} 个")
        print(f"   语义数据: {upload_info.get('total_semantic_uploads', 0)} 个")
        print(f"   向量数据: {upload_info.get('total_vector_uploads', 0)} 个")
        print(f"   错误数量: {upload_info['total_errors']}")
        
        # 质量指标
        quality_info = summary['quality_metrics']
        print(f"\n📈 质量指标:")
        print(f"   角色覆盖率: {quality_info['classification_coverage']['role_coverage_percent']}%")
        print(f"   行业覆盖率: {quality_info['classification_coverage']['industry_coverage_percent']}%")
        print(f"   平均角色技能: {quality_info['baseline_quality']['avg_skills_per_role']}")
        print(f"   平均行业技能: {quality_info['baseline_quality']['avg_skills_per_industry']}")
        
        print(f"\n📁 输出目录: {self.config.output_dir}")
        print("="*60)

class EndToEndProcessor:
    """端到端处理器 - 主要接口类"""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        初始化端到端处理器
        
        Args:
            config_path: 配置文件路径（可选）
            **kwargs: 配置参数覆盖
        """
        self.config = self._load_config(config_path, **kwargs)
        self.orchestrator = PipelineOrchestrator(self.config)
    
    def _load_config(self, config_path: Optional[str], **kwargs) -> PipelineConfig:
        """加载配置"""
        base_config = {}
        
        # 从配置文件加载
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                base_config = json.load(f)
            logger.info(f"✅ 从文件加载配置: {config_path}")
        
        # 应用参数覆盖
        base_config.update(kwargs)
        
        return PipelineConfig(**base_config)
    
    def process_vocabulary(self, vocab_path: str) -> Dict:
        """处理技能词汇表"""
        skills_vocab = self.orchestrator._load_vocabulary(vocab_path)
        all_skills = self.orchestrator.processor.extract_all_skills(skills_vocab)
        return {
            'vocabulary': skills_vocab,
            'skills': all_skills,
            'total_skills': len(all_skills)
        }
    
    def generate_classifications(self, all_skills: List[Dict]) -> Tuple[Dict, Dict]:
        """生成技能分类"""
        return self.orchestrator.classifier.classify_skills(all_skills)
    
    def generate_baselines(self, role_classifications: Dict, industry_classifications: Dict) -> Dict:
        """生成语义基线"""
        return self.orchestrator.baseline_generator.generate_all_baselines(
            role_classifications, industry_classifications
        )
    
    def upload_to_database(self, baselines: Dict) -> Dict:
        """上传基线到数据库"""
        return self.orchestrator.database_manager.upload_baselines(baselines)
    
    def run_complete_pipeline(self, vocab_path: str) -> Dict:
        """运行完整管线"""
        return self.orchestrator.run_complete_pipeline(vocab_path)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='JobbAI 端到端技能处理管线')
    
    # 输入输出参数
    parser.add_argument('vocabulary', help='技能词汇表文件路径')
    parser.add_argument('--output-dir', '-o', default='output', help='输出目录')
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--model-name', help='SentenceTransformer模型名称')
    
    # 分级阈值参数
    parser.add_argument('--high-confidence', type=float, default=0.75, help='高置信度阈值')
    parser.add_argument('--medium-confidence', type=float, default=0.55, help='中等置信度阈值') 
    parser.add_argument('--low-confidence', type=float, default=0.35, help='低置信度阈值')
    
    # 兼容旧版本参数
    parser.add_argument('--role-threshold', type=float, help='角色分配阈值（兼容旧版本）')
    parser.add_argument('--industry-threshold', type=float, help='行业分配阈值（兼容旧版本）')
    
    # 调试选项
    parser.add_argument('--enable-debug', action='store_true', help='启用调试日志')
    parser.add_argument('--save-details', action='store_true', default=True, help='保存分类详情')
    
    # 数据库参数
    parser.add_argument('--supabase-url', help='Supabase URL')
    parser.add_argument('--supabase-key', help='Supabase API Key')
    parser.add_argument('--skip-database', action='store_true', help='跳过数据库上传')
    
    # 其他选项
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 准备配置参数
    config_kwargs = {
        'output_dir': args.output_dir
    }
    
    if args.model_name:
        config_kwargs['model_name'] = args.model_name
    if args.high_confidence is not None:
        config_kwargs['high_confidence_threshold'] = args.high_confidence
    if args.medium_confidence is not None:
        config_kwargs['medium_confidence_threshold'] = args.medium_confidence  
    if args.low_confidence is not None:
        config_kwargs['low_confidence_threshold'] = args.low_confidence
        
    # 兼容旧版本参数
    if args.role_threshold is not None:
        config_kwargs['role_threshold'] = args.role_threshold
    if args.industry_threshold is not None:
        config_kwargs['industry_threshold'] = args.industry_threshold
        
    if args.supabase_url:
        config_kwargs['supabase_url'] = args.supabase_url
    if args.supabase_key:
        config_kwargs['supabase_key'] = args.supabase_key
    if args.skip_database:
        config_kwargs['supabase_url'] = None
        config_kwargs['supabase_key'] = None
    if args.enable_debug:
        config_kwargs['enable_debug_logging'] = True
    if args.save_details is not None:
        config_kwargs['save_classification_details'] = args.save_details
    
    print("🚀 JobbAI 端到端技能处理管线")
    print(f"📁 词汇表文件: {args.vocabulary}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🔧 配置文件: {args.config or '使用默认配置'}")
    print("-" * 60)
    
    try:
        # 初始化处理器
        processor = EndToEndProcessor(args.config, **config_kwargs)
        
        # 运行完整管线
        results = processor.run_complete_pipeline(args.vocabulary)
        
        print("\n🎉 管线执行成功完成!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 管线执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main())