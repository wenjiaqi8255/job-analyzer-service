import logging
from typing import Dict, List, Tuple, Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .config import PipelineConfig

logger = logging.getLogger(__name__)


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
        all_role_keys = set(self.explicit_role_mappings.keys()) | set(self.role_semantic_descriptions.keys())
        role_classifications = {role: {
            'essential': [], 'common': [], 'collaboration': [], 'specializations': []
        } for role in all_role_keys}

        all_industry_keys = set(self.explicit_industry_mappings.keys()) | set(self.industry_semantic_descriptions.keys())
        industry_classifications = {industry: {
            'core_domain': [], 'regulatory': [], 'business_focus': [], 'unique_requirements': []
        } for industry in all_industry_keys}

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
                if best_role not in role_classifications:
                    logger.warning(f"Semantic match found for role '{best_role}' but it's not in the classification dictionary. Skipping.")
                    continue
                category = self._determine_role_category(skill_info, role_similarity)
                role_classifications[best_role][category].append(skill)
                assigned_skills.add(skill)
                stats['semantic_role_assigned'] += 1
                continue

            # 高置信度行业匹配
            best_industry, industry_similarity = self._calculate_best_industry_match(skill_info)
            if best_industry and industry_similarity >= self.high_confidence_threshold:
                if best_industry not in industry_classifications:
                    logger.warning(f"Semantic match found for industry '{best_industry}' but it's not in the classification dictionary. Skipping.")
                    continue
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
                if best_role not in role_classifications:
                    logger.warning(f"Semantic match found for role '{best_role}' but it's not in the classification dictionary. Skipping.")
                    continue
                category = self._determine_role_category(skill_info, role_similarity)
                role_classifications[best_role][category].append(skill)
                assigned_skills.add(skill)
                stats['semantic_role_assigned'] += 1
                continue

            # 中等置信度行业匹配
            best_industry, industry_similarity = self._calculate_best_industry_match(skill_info)
            if best_industry and industry_similarity >= self.medium_confidence_threshold:
                if best_industry not in industry_classifications:
                    logger.warning(f"Semantic match found for industry '{best_industry}' but it's not in the classification dictionary. Skipping.")
                    continue
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
            if best_role and role_similarity >= self.low_confidence_threshold and role_similarity > industry_similarity:
                if best_role not in role_classifications:
                    logger.warning(f"Semantic match found for role '{best_role}' but it's not in the classification dictionary. Skipping.")
                else:
                    category = self._determine_role_category(skill_info, role_similarity)
                    role_classifications[best_role][category].append(skill)
                    stats['semantic_role_assigned'] += 1
            elif best_industry and industry_similarity >= self.low_confidence_threshold:
                if best_industry not in industry_classifications:
                    logger.warning(f"Semantic match found for industry '{best_industry}' but it's not in the classification dictionary. Skipping.")
                else:
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

        if not self.role_embeddings:
            return None, 0.0

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

        if not self.industry_embeddings:
            return None, 0.0

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
