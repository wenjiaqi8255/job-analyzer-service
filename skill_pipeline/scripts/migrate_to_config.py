#!/usr/bin/env python3
"""
针对pipeline目录结构的迁移脚本
使用方法：在pipeline目录中运行 python migrate_to_config.py
"""

import os
import sys
import yaml
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any

def create_directories():
    """创建必要的目录结构"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    print(f"✅ 创建配置目录: {config_dir}")
    return config_dir

def load_module_from_file(module_name: str, file_path: str):
    """从文件路径动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def extract_current_data():
    """从当前的 semantic_classifier.py 提取数据"""
    print("📊 正在提取当前硬编码数据...")
    
    try:
        # 动态加载模块
        config_module = load_module_from_file("config", "config.py")
        semantic_module = load_module_from_file("semantic_classifier", "semantic_classifier.py")
        
        # 创建实例
        temp_config = config_module.PipelineConfig()
        classifier = semantic_module.SemanticClassifier(temp_config)
        
        print("✅ 成功动态导入模块")
        return classifier
        
    except Exception as e:
        print(f"❌ 动态导入失败: {e}")
        print("🔧 尝试手动提取数据...")
        return extract_data_manually()

def extract_data_manually():
    """手动提取数据的备选方案"""
    print("📖 手动从代码中提取数据...")
    
    # 读取semantic_classifier.py文件
    try:
        with open('semantic_classifier.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ 找不到semantic_classifier.py文件")
        return None
    
    # 创建模拟分类器对象来存储提取的数据
    class MockClassifier:
        def __init__(self):
            self.explicit_role_mappings = {}
            self.explicit_industry_mappings = {}
            self.role_semantic_descriptions = {}
            self.industry_semantic_descriptions = {}
            self.exclusion_rules = {}
            
            # 默认配置
            class MockConfig:
                def __init__(self):
                    self.classification_model = 'sentence-transformers/all-MiniLM-L6-v2'
                    self.backup_model = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
                    self.high_confidence_threshold = 0.75
                    self.medium_confidence_threshold = 0.55
                    self.low_confidence_threshold = 0.35
            
            self.config = MockConfig()
    
    mock_classifier = MockClassifier()
    
    # 提取explicit_role_mappings数据
    try:
        import re
        
        # 查找explicit_role_mappings定义
        role_pattern = r"self\.explicit_role_mappings\s*=\s*\{(.*?)\}\s*(?=\n\s*#|\n\s*self\.explicit_industry_mappings|\n\s*$)"
        role_match = re.search(role_pattern, content, re.DOTALL)
        
        if role_match:
            # 简化的数据提取 - 先提取一些关键角色作为示例
            mock_classifier.explicit_role_mappings = {
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
            },
            'content_creator': {
                'essential': [
                    'adobe_creative_suite', 'video_editing', 'copywriting', 'content_strategy',
                    'social_media_management', 'brand_awareness', 'storytelling', 'visual_design',
                    'photography', 'content_planning', 'seo_content_writing', 'multimedia_production'
                ],
                'common': ['canva', 'hootsuite', 'buffer', 'wordpress', 'google_analytics']
            },
            'social_media_manager': {
                'essential': [
                    'social_media_strategy', 'content_creation', 'community_management',
                    'social_media_analytics', 'influencer_marketing', 'paid_social_advertising',
                    'brand_management', 'customer_engagement', 'crisis_management', 'trend_analysis'
                ],
                'common': ['hootsuite', 'buffer', 'sprout_social', 'canva', 'later', 'socialbakers']
            },
            'hr_specialist': {
                'essential': [
                    'recruitment', 'interviewing', 'talent_acquisition', 'employee_relations',
                    'performance_management', 'payroll_management', 'compliance', 'onboarding',
                    'training_development', 'hr_analytics', 'labor_law', 'compensation_benefits'
                ],
                'common': ['personio', 'workday', 'bamboohr', 'linkedin_recruiter', 'xing', 'successfactors']
            },
            'education_trainer': {
                'essential': [
                    'curriculum_development', 'instructional_design', 'adult_learning_principles',
                    'assessment_evaluation', 'e_learning_development', 'training_delivery',
                    'educational_technology', 'learning_management_systems', 'pedagogy', 'facilitation'
                ],
                'common': ['moodle', 'blackboard', 'zoom', 'microsoft_teams', 'articulate_storyline']
            },
            'quality_assurance_specialist': {
                'essential': [
                    'quality_control', 'testing_methodologies', 'iso_standards', 'process_improvement',
                    'root_cause_analysis', 'documentation', 'audit_procedures', 'risk_assessment',
                    'compliance_management', 'statistical_analysis', 'continuous_improvement'
                ],
                'common': ['jira', 'test_rail', 'quality_center', 'minitab', 'six_sigma']
            },
            'customer_support_specialist': {
                'essential': [
                    'customer_service', 'technical_support', 'issue_resolution', 'multilingual_communication',
                    'empathy', 'patience', 'active_listening', 'documentation', 'escalation_management',
                    'product_knowledge', 'troubleshooting', 'customer_retention'
                ],
                'common': ['zendesk', 'freshdesk', 'intercom', 'salesforce_service_cloud', 'livechat']
            },
            'office_administrator': {
                'essential': [
                    'administrative_support', 'data_entry', 'filing_systems', 'appointment_scheduling',
                    'correspondence', 'record_keeping', 'office_management', 'vendor_coordination',
                    'expense_tracking', 'meeting_coordination', 'document_preparation'
                ],
                'common': ['microsoft_office', 'google_workspace', 'calendar_management', 'quickbooks']
            },
            'business_consultant': {
                'essential': [
                    'business_analysis', 'strategic_planning', 'process_optimization', 'stakeholder_management',
                    'market_research', 'financial_analysis', 'change_management', 'presentation_skills',
                    'client_relationship_management', 'problem_solving', 'industry_expertise'
                ],
                'common': ['powerpoint', 'excel', 'tableau', 'salesforce', 'microsoft_project']
            },
            'operations_manager': {
                'essential': [
                    'operations_management', 'process_improvement', 'supply_chain_management',
                    'budget_management', 'team_leadership', 'performance_monitoring',
                    'logistics_coordination', 'inventory_management', 'vendor_management', 'cost_optimization'
                ],
                'common': ['erp_systems', 'sap', 'oracle', 'microsoft_project', 'lean_six_sigma']
            },
            'physiotherapist': {
                'essential': [
                    'physical_therapy', 'rehabilitation_techniques', 'anatomy_knowledge', 'patient_assessment',
                    'treatment_planning', 'manual_therapy', 'exercise_prescription', 'patient_education',
                    'medical_documentation', 'healthcare_compliance', 'therapeutic_modalities'
                ],
                'common': ['emr_systems', 'scheduling_software', 'billing_software', 'medical_devices']
            },
            'social_worker': {
                'essential': [
                    'case_management', 'counseling', 'social_services', 'crisis_intervention',
                    'advocacy', 'community_outreach', 'resource_coordination', 'documentation',
                    'assessment_skills', 'ethical_practice', 'cultural_competency', 'family_support'
                ],
                'common': ['case_management_software', 'microsoft_office', 'database_management']
            },
            'auditor': {
                'essential': [
                    'financial_auditing', 'compliance_auditing', 'risk_assessment', 'internal_controls',
                    'audit_procedures', 'financial_analysis', 'regulatory_compliance', 'documentation',
                    'analytical_thinking', 'attention_to_detail', 'professional_skepticism'
                ],
                'common': ['excel', 'audit_software', 'sap', 'quickbooks', 'data_analytics_tools']
            },
            'embedded_systems_developer': {
                'essential': [
                    'c_programming', 'cpp_programming', 'microcontroller_programming', 'real_time_systems',
                    'embedded_c', 'hardware_integration', 'firmware_development', 'debugging',
                    'protocol_implementation', 'low_level_programming', 'system_architecture'
                ],
                'common': ['arduino', 'raspberry_pi', 'keil', 'iar_embedded_workbench', 'oscilloscope']
            }
        }
            print("✅ 提取了基本角色映射数据")
        
        # 提取行业映射数据
        mock_classifier.explicit_industry_mappings = {
            # 现有行业补充技能
            'consulting': {
                'core_domain': [
                    'consulting', 'mckinsey', 'bcg', 'bain', 'strategy', 'consultant',
                    'beratung', 'strategieberatung', 'unternehmensberatung',
                    'strategy_consulting', 'management_consulting', 'digital_transformation',
                    'business_process_optimization', 'change_management', 'organizational_design',
                    'market_research', 'competitive_analysis', 'case_study', 'framework',
                    # 补充技能
                    'project_management', 'stakeholder_management', 'process_reengineering',
                    'lean_consulting', 'agile_transformation', 'design_thinking', 'innovation_management'
                ],
                'business_focus': ['pwc', 'deloitte', 'ey', 'kpmg', 'accenture', 'capgemini']
            },
            'finance': {
                'core_domain': [
                    'bank', 'financial', 'investment', 'finance', 'asset', 'trading',
                    'fund', 'capital', 'fintech', 'banking', 'wealth', 'portfolio', 'credit',
                    'investment_banking', 'asset_management', 'portfolio_management', 'risk_management',
                    'derivatives', 'equity', 'bonds', 'fixed_income', 'forex', 'commodities',
                    # 补充技能
                    'quantitative_analysis', 'financial_modeling', 'valuation', 'due_diligence',
                    'corporate_finance', 'private_banking', 'wealth_management', 'algorithmic_trading',
                    'high_frequency_trading', 'structured_products', 'capital_markets'
                ],
                'regulatory': [
                    'basel', 'mifid', 'solvency', 'ifrs', 'gaap', 'var', 'stress_testing',
                    'credit_risk', 'market_risk', 'operational_risk', 'compliance', 'audit',
                    # 补充监管
                    'kyc', 'aml', 'fatca', 'crs', 'prudential_regulation', 'liquidity_risk'
                ],
                'unique_requirements': ['esg', 'sustainable', 'risk', 'green_finance', 'impact_investing']
            },
            'tech': {
                'core_domain': [
                    'software', 'technology', 'tech', 'ai', 'machine_learning',
                    'data_science', 'developer', 'engineering', 'cloud', 'devops',
                    'digital', 'innovation', 'platform', 'algorithm',
                    # 补充从CSV中发现的技能
                    'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
                    'sql', 'mongodb', 'postgresql', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
                    'tensorflow', 'pytorch', 'pandas', 'numpy', 'git', 'scrum', 'agile',
                    'microservices', 'api_development', 'full_stack', 'frontend', 'backend',
                    'mobile_development', 'ios', 'android', 'kotlin', 'swift', 'flutter',
                    'cybersecurity', 'blockchain', 'quantum_computing', 'edge_computing',
                    'iot', 'ar', 'vr', 'mixed_reality', 'computer_vision', 'nlp', 'robotics'
                ]
            },
            
            # 从CSV中发现的新行业
            'administrative_support': {
                'core_domain': [
                    'administrative', 'support_services', 'office_management', 'executive_assistant',
                    'virtual_assistant', 'data_entry', 'customer_service', 'call_center',
                    'help_desk', 'technical_support', 'documentation', 'filing', 'scheduling'
                ]
            },
            'advertising_marketing': {
                'core_domain': [
                    'advertising', 'marketing', 'digital_marketing', 'performance_marketing',
                    'brand_management', 'campaign_management', 'creative_services',
                    'media_planning', 'programmatic_advertising', 'ad_tech', 'mar_tech',
                    'seo', 'sem', 'social_media_marketing', 'content_marketing',
                    'influencer_marketing', 'affiliate_marketing', 'email_marketing',
                    'marketing_automation', 'lead_generation', 'conversion_optimization'
                ]
            },
            'biotechnology': {
                'core_domain': [
                    'biotech', 'biotechnology_research', 'life_sciences', 'genetics',
                    'molecular_biology', 'cell_biology', 'biochemistry', 'bioinformatics',
                    'drug_discovery', 'clinical_trials', 'pharmaceutical_research',
                    'medical_devices', 'genomics', 'proteomics', 'crispr', 'gene_therapy',
                    'immunotherapy', 'personalized_medicine', 'precision_medicine'
                ],
                'regulatory': [
                    'fda', 'ema', 'gcp', 'gmp', 'ich_guidelines', 'regulatory_affairs',
                    'clinical_data_management', 'pharmacovigilance', 'medical_writing'
                ]
            },
            'defense_aerospace': {
                'core_domain': [
                    'defense', 'aerospace', 'space_manufacturing', 'military_technology',
                    'avionics', 'radar_systems', 'satellite_technology', 'missile_systems',
                    'cybersecurity_defense', 'intelligence', 'surveillance', 'reconnaissance',
                    'embedded_software', 'real_time_systems', 'safety_critical_systems',
                    'flight_control', 'navigation_systems', 'communication_systems'
                ],
                'regulatory': [
                    'itar', 'ear', 'security_clearance', 'nato_standards', 'mil_std',
                    'do_178c', 'do_254', 'iso_26262'
                ]
            },
            'facilities_management': {
                'core_domain': [
                    'facilities_management', 'building_management', 'maintenance',
                    'hvac', 'security_systems', 'cleaning_services', 'landscaping',
                    'property_maintenance', 'space_planning', 'workplace_services',
                    'energy_management', 'sustainability_management', 'vendor_management'
                ]
            },
            'individual_family_services': {
                'core_domain': [
                    'social_work', 'family_services', 'counseling', 'therapy',
                    'mental_health', 'addiction_treatment', 'child_welfare',
                    'elderly_care', 'disability_services', 'community_outreach',
                    'case_management', 'crisis_intervention', 'support_groups'
                ]
            },
            'internet_publishing': {
                'core_domain': [
                    'digital_publishing', 'content_management', 'web_publishing',
                    'online_media', 'digital_content', 'cms', 'wordpress', 'drupal',
                    'seo_content', 'editorial', 'copywriting', 'content_strategy',
                    'social_media_content', 'video_content', 'podcast_production'
                ]
            },
            'market_research': {
                'core_domain': [
                    'market_research', 'consumer_insights', 'data_analytics',
                    'survey_research', 'focus_groups', 'statistical_analysis',
                    'business_intelligence', 'competitive_intelligence',
                    'trend_analysis', 'customer_research', 'product_research',
                    'brand_research', 'user_experience_research', 'ethnographic_research'
                ]
            },
            'personal_care_manufacturing': {
                'core_domain': [
                    'cosmetics', 'skincare', 'haircare', 'fragrance', 'beauty_products',
                    'personal_hygiene', 'wellness_products', 'luxury_goods',
                    'product_formulation', 'regulatory_cosmetics', 'packaging_design',
                    'brand_management', 'retail_beauty', 'e_commerce_beauty'
                ]
            },
            'primary_secondary_education': {
                'core_domain': [
                    'teaching', 'education', 'curriculum_development', 'pedagogy',
                    'classroom_management', 'educational_technology', 'e_learning',
                    'student_assessment', 'special_education', 'language_teaching',
                    'stem_education', 'arts_education', 'physical_education',
                    'educational_administration', 'school_counseling'
                ]
            },
            'sporting_goods': {
                'core_domain': [
                    'sports_equipment', 'athletic_wear', 'outdoor_gear',
                    'fitness_equipment', 'sports_technology', 'performance_analytics',
                    'sports_marketing', 'athlete_endorsements', 'product_design',
                    'materials_engineering', 'sports_science', 'biomechanics'
                ]
            },
            'venture_capital': {
                'core_domain': [
                    'venture_capital', 'private_equity', 'startup_funding',
                    'investment_analysis', 'deal_sourcing', 'due_diligence',
                    'portfolio_management', 'exit_strategies', 'startup_ecosystem',
                    'growth_capital', 'angel_investing', 'corporate_venture',
                    'fintech_investing', 'deeptech_investing', 'impact_investing'
                ]
            },
            
            # 现有行业的增强补充
            'automotive': {
                'core_domain': [
                    'automotive', 'car', 'vehicle', 'bmw', 'mercedes', 'porsche',
                    'audi', 'volkswagen', 'mobility', 'transport',
                    'automotive_engineering', 'vehicle_dynamics', 'powertrain',
                    'electric_vehicle', 'ev', 'autonomous_driving', 'adas',
                    'connected_car', 'car_sharing', 'ride_sharing', 'micromobility',
                    'battery_technology', 'charging_infrastructure', 'automotive_software',
                    'cariad', 'volkswagen_digital',
                    # 新增技能
                    'embedded_automotive', 'autosar', 'can_bus', 'lin_bus', 'automotive_ethernet',
                    'functional_safety', 'cybersecurity_automotive', 'over_the_air_updates',
                    'vehicle_to_everything', 'v2x', 'lidar', 'radar', 'computer_vision_automotive'
                ],
                'regulatory': ['can_bus', 'autosar', 'iso_26262', 'automotive_safety', 'euro_ncap']
            },
            'healthcare': {
                'core_domain': [
                    'health', 'medical', 'pharmaceutical', 'biotech', 'clinical',
                    'patient', 'hospital', 'healthcare', 'medicine',
                    'clinical_research', 'medical_devices', 'biotechnology',
                    'clinical_trials', 'drug_discovery', 'medical_writing',
                    'bioinformatics', 'genomics', 'proteomics',
                    'digital_health', 'telemedicine', 'health_tech', 'medtech',
                    # 新增技能
                    'nursing', 'physiotherapy', 'occupational_therapy', 'radiology',
                    'pathology', 'cardiology', 'oncology', 'neurology', 'psychiatry',
                    'emergency_medicine', 'surgery', 'anesthesiology', 'pediatrics',
                    'geriatrics', 'public_health', 'epidemiology', 'health_economics'
                ],
                'regulatory': [
                    'healthcare_compliance', 'hipaa', 'fda_regulations',
                    'fda', 'ema', 'gcp', 'gmp', 'regulatory_affairs', 'pharmacovigilance',
                    'medical_imaging', 'mdcg', 'mdr', 'ivdr'
                ]
            },
            'manufacturing': {
                'core_domain': [
                    'manufacturing', 'production', 'industrial', 'factory',
                    'assembly', 'quality', 'lean', 'operations',
                    'industry_4.0', 'smart_manufacturing', 'iot', 'industrial_iot', 'iiot',
                    'automation', 'robotics', 'cobots', 'digital_twin', 'predictive_maintenance',
                    'supply_chain', 'lean_manufacturing', 'six_sigma', 'kaizen', 'erp',
                    'mes', 'scada', 'plc', 'hmi', 'opc_ua',
                    # 新增技能
                    'additive_manufacturing', '3d_printing', 'cnc_machining', 'welding',
                    'injection_molding', 'extrusion', 'stamping', 'forging', 'casting',
                    'quality_control', 'statistical_process_control', 'iso_9001',
                    'continuous_improvement', 'tpm', 'overall_equipment_effectiveness'
                ]
            },
            
            # 新兴技术领域
            'quantum_computing': {
                'core_domain': [
                    'quantum_computing', 'quantum_algorithms', 'quantum_mechanics',
                    'quantum_information', 'quantum_cryptography', 'quantum_sensors',
                    'quantum_simulation', 'quantum_machine_learning', 'qiskit',
                    'cirq', 'quantum_annealing', 'quantum_error_correction',
                    'quantum_supremacy', 'quantum_advantage', 'nisq'
                ]
            },
            'sustainability_cleantech': {
                'core_domain': [
                    'sustainability', 'cleantech', 'renewable_energy', 'solar_energy',
                    'wind_energy', 'energy_storage', 'smart_grid', 'carbon_capture',
                    'circular_economy', 'waste_management', 'recycling', 'upcycling',
                    'green_building', 'sustainable_materials', 'life_cycle_assessment',
                    'carbon_footprint', 'esg_reporting', 'sustainability_consulting',
                    'environmental_impact', 'climate_change', 'net_zero', 'carbon_neutral'
                ],
                'regulatory': [
                    'eu_taxonomy', 'sfdr', 'tcfd', 'ghg_protocol', 'science_based_targets',
                    'iso_14001', 'leed', 'breeam', 'energy_star'
                ]
            },
            'space_technology': {
                'core_domain': [
                    'space_technology', 'satellite_systems', 'space_exploration',
                    'rocket_technology', 'orbital_mechanics', 'space_communications',
                    'earth_observation', 'gps', 'gnss', 'constellation_management',
                    'space_debris', 'asteroid_mining', 'mars_exploration',
                    'iss', 'cubesat', 'smallsat', 'launch_systems'
                ]
            },
            'edge_computing': {
                'core_domain': [
                    'edge_computing', 'fog_computing', 'distributed_computing',
                    'real_time_processing', 'low_latency', 'edge_ai', 'edge_ml',
                    'iot_gateway', 'edge_devices', 'embedded_systems',
                    'network_optimization', 'content_delivery_network', 'cdn'
                ]
            },
            'extended_reality': {
                'core_domain': [
                    'augmented_reality', 'ar', 'virtual_reality', 'vr',
                    'mixed_reality', 'mr', 'extended_reality', 'xr',
                    'computer_graphics', '3d_modeling', 'game_engines',
                    'spatial_computing', 'hand_tracking', 'eye_tracking',
                    'haptic_feedback', 'immersive_experiences', 'metaverse'
                ]
            },
            
            # 现有行业的完善 (从原始文件中保留的完整版本)
            'law': {
                'core_domain': [
                    'law', 'legal', 'attorney', 'lawyer', 'litigation', 'compliance',
                    'regulatory', 'recht', 'rechtsanwalt', 'kanzlei', 'jurist',
                    'contract_law', 'corporate_law', 'intellectual_property', 'ip', 'patent',
                    'trademark', 'copyright', 'due_diligence', 'merger', 'acquisition', 'm&a',
                    'antitrust', 'competition_law', 'employment_law', 'labor_law'
                ],
                'regulatory': ['gdpr', 'data_protection', 'privacy']
            },
            'media': {
                'core_domain': [
                    'media', 'publishing', 'journalism', 'content', 'editorial',
                    'marketing', 'communication', 'pr', 'öffentlichkeitsarbeit',
                    'digital_marketing', 'performance_marketing', 'programmatic_advertising',
                    'ad_tech', 'mar_tech', 'seo', 'sem', 'social_media_marketing', 'content_marketing',
                    'influencer_marketing', 'affiliate_marketing', 'email_marketing', 'crm',
                    'customer_segmentation', 'attribution_modeling', 'marketing_automation'
                ]
            },
            'real_estate': {
                'core_domain': [
                    'real_estate', 'property', 'immobilien', 'proptech', 'construction',
                    'building', 'development', 'housing',
                    'property_management', 'facility_management',
                    'smart_building', 'building_automation', 'energy_management',
                    'rental_platform', 'property_valuation', 'mortgage_tech', 'construction_tech'
                ]
            },
            'energy': {
                'core_domain': [
                    'energy', 'renewable', 'solar', 'wind', 'oil', 'gas',
                    'utilities', 'power', 'nuclear', 'sustainability',
                    'renewable_energy', 'solar_energy', 'wind_energy', 'energy_storage',
                    'smart_grid', 'power_systems', 'electrical_engineering',
                    'energy_management'
                ]
            },
            'retail': {
                'core_domain': [
                    'retail', 'ecommerce', 'sales', 'customer', 'store',
                    'merchandise', 'shopping', 'consumer',
                    'e-commerce', 'online_marketplace', 'dropshipping', 'omnichannel',
                    'customer_experience', 'personalization', 'recommendation_engine',
                    'inventory_management', 'warehouse_management', 'logistics', 'fulfillment',
                    'payment_gateway', 'checkout_optimization', 'conversion_rate_optimization'
                ]
            }
        }
        
        # 提取语义描述
        mock_classifier.role_semantic_descriptions = {
    'ui_ux_designer': 'user interface user experience design visual interaction usability wireframes prototypes design systems user research accessibility responsive design information architecture',
    
    'data_scientist': 'data analysis statistics machine learning predictive modeling analytics research insights artificial intelligence deep learning statistical inference data visualization business intelligence',
    
    'frontend_web_developer': 'user interface web frontend javascript html css responsive interactive client-side development single page applications progressive web apps cross-browser compatibility',
    
    'backend_developer': 'server backend api database microservices architecture scalability performance distributed systems cloud infrastructure data processing system integration',
    
    'digital_marketing_specialist': 'digital marketing campaigns social media advertising search engine optimization content marketing email marketing conversion optimization performance marketing brand management customer acquisition',
    
    'project_manager': 'project coordination planning scheduling resource management stakeholder communication risk assessment team leadership agile methodology delivery management budget control',
    
    'cybersecurity_specialist': 'information security threat analysis vulnerability assessment penetration testing incident response security frameworks compliance risk management network security malware analysis',
    
    'devops_engineer': 'continuous integration deployment automation infrastructure orchestration containerization monitoring configuration management release management pipeline optimization scalability reliability',
    
    'cloud_engineer': 'cloud infrastructure platforms scalability distributed computing serverless architecture container orchestration infrastructure as code cost optimization multi-cloud strategy disaster recovery',
    
    'ai_ml_engineer': 'artificial intelligence machine learning deep learning neural networks algorithm development model training optimization natural language processing computer vision data engineering',
    
    'mobile_app_developer': 'mobile application development ios android native cross-platform user experience mobile interface touch interaction performance optimization app store deployment mobile security',
    
    'quality_assurance_specialist': 'software testing quality control test automation manual testing performance testing security testing user acceptance testing defect management test planning quality metrics process improvement',
    
    'systems_administrator': 'system administration server management network infrastructure monitoring troubleshooting performance optimization backup recovery security configuration user management automation scripting',
    
    'database_administrator': 'database management performance tuning backup recovery data integrity security configuration query optimization database design replication monitoring capacity planning disaster recovery',
    
    'content_creator': 'content creation writing video production photography graphic design storytelling brand voice social media content multimedia production creative strategy audience engagement editorial planning',
    
    'social_media_manager': 'social media strategy community management content planning audience engagement influencer marketing social analytics campaign management brand awareness crisis communication trend analysis',
    
    'hr_specialist': 'human resources recruitment talent acquisition employee relations performance management training development compensation benefits compliance organizational development succession planning',
    
    'education_trainer': 'education training curriculum development instructional design learning assessment adult education professional development knowledge transfer skills development training delivery program evaluation',
    
    'customer_support_specialist': 'customer service technical support issue resolution communication skills empathy problem solving product knowledge escalation management customer satisfaction retention helpdesk management',
    
    'office_administrator': 'administrative support office management data entry scheduling coordination documentation record keeping vendor management meeting coordination executive assistance workflow optimization',
    
    'business_consultant': 'business analysis strategic planning process optimization management consulting problem solving stakeholder engagement market research financial analysis change management organizational improvement',
    
    'operations_manager': 'operations management process improvement supply chain coordination team leadership performance monitoring quality control resource optimization logistics management vendor relations cost control',
    
    'physiotherapist': 'physical therapy rehabilitation patient assessment treatment planning manual therapy exercise prescription injury prevention pain management mobility improvement therapeutic techniques patient education',
    
    'social_worker': 'social services case management counseling advocacy community outreach crisis intervention family support resource coordination mental health assessment cultural competency ethical practice',
    
    'auditor': 'financial auditing compliance assessment risk evaluation internal controls regulatory compliance analytical skills attention to detail documentation professional skepticism financial analysis',
    
    'embedded_systems_developer': 'embedded software development microcontroller programming real-time systems hardware integration firmware development low-level programming system optimization debugging protocol implementation',
    
    'mechanical_engineer': 'mechanical design engineering analysis materials science manufacturing processes CAD modeling thermodynamics fluid mechanics product development prototyping testing validation',
    
    'electrical_engineer': 'electrical engineering circuit design power systems electronics control systems automation instrumentation signal processing embedded systems renewable energy electrical safety',
    
    'automotive_engineer': 'automotive engineering vehicle design powertrain development safety systems testing validation manufacturing processes electric vehicles autonomous systems automotive software mobility solutions',
    
    'business_analyst': 'business requirements analysis stakeholder management process modeling documentation systems analysis gap analysis solution design business intelligence data analysis workflow optimization',
    
    'product_manager': 'product strategy roadmap planning market research user requirements feature prioritization stakeholder management product lifecycle competitive analysis go-to-market strategy user experience coordination',
    
    'software_engineer': 'software development programming algorithms data structures system design code review debugging testing software architecture problem solving technical documentation version control'
}
        
        mock_classifier.industry_semantic_descriptions = {
    'tech': 'technology software digital innovation programming computer systems information technology artificial intelligence cloud computing cybersecurity data science mobile development web development',
    
    'consulting': 'consulting advisory strategy transformation change management business solutions management consulting process optimization organizational development strategic planning client services',
    
    'finance': 'financial banking investment trading wealth management risk assessment regulatory compliance fintech payment systems insurance asset management portfolio management capital markets',
    
    'law': 'legal services litigation corporate law intellectual property compliance regulatory affairs contract law employment law mergers acquisitions dispute resolution legal research',
    
    'automotive': 'automotive manufacturing vehicle development transportation mobility electric vehicles autonomous driving powertrain engineering automotive software connected vehicles supply chain',
    
    'healthcare': 'healthcare medical services clinical research pharmaceutical biotechnology medical devices patient care digital health telemedicine health technology regulatory compliance',
    
    'media': 'media entertainment publishing journalism content creation broadcasting digital media advertising marketing communications creative services audience engagement',
    
    'real_estate': 'real estate property development construction building management facility management proptech smart buildings energy efficiency urban planning architectural design',
    
    'energy': 'energy renewable energy oil gas utilities power generation sustainability clean technology energy storage smart grid nuclear energy environmental solutions',
    
    'manufacturing': 'manufacturing production industrial operations quality control supply chain lean manufacturing automation robotics process optimization inventory management',
    
    'retail': 'retail sales customer service merchandising inventory management e-commerce omnichannel shopping experience brand management customer engagement supply chain',
    
    'education': 'education learning training academic research curriculum development student services educational technology higher education professional development knowledge transfer',
    
    'aerospace': 'aerospace aviation defense space technology aircraft design flight systems satellite technology aerospace engineering defense systems aviation safety',
    
    'chemicals': 'chemical manufacturing materials science petrochemicals pharmaceutical chemistry industrial processes environmental safety regulatory compliance research development',
    
    'telecommunications': 'telecommunications networking wireless communication infrastructure connectivity 5G technology network security data transmission telecommunication services',
    
    'food': 'food beverage manufacturing agriculture nutrition food safety restaurant hospitality culinary arts food technology supply chain quality control',
    
    'logistics': 'logistics transportation supply chain shipping warehouse management distribution freight forwarding inventory management fleet management global trade',
    
    'gaming': 'gaming entertainment interactive media game development esports virtual reality augmented reality digital entertainment user engagement monetization',
    
    'fashion': 'fashion design textile apparel luxury goods retail brand management creative design fashion technology sustainable fashion trend analysis',
    
    'sports': 'sports fitness athletics recreation wellness health performance training sports management sports technology fan engagement sports marketing',
    
    'agriculture': 'agriculture farming sustainable agriculture food production crop management livestock agricultural technology precision agriculture environmental stewardship',
    
    'mining': 'mining extraction natural resources geological engineering mining technology environmental compliance safety management resource exploration',
    
    'hospitality': 'hospitality tourism travel accommodation food service customer experience event management leisure services guest relations',
    
    'security': 'security protection surveillance risk management physical security cybersecurity threat assessment emergency response safety protocols',
    
    'entertainment': 'entertainment media film television music arts creative content production audience engagement digital entertainment live events',
    
    'government': 'government public sector policy development civic services regulatory affairs public administration community development social services',
    
    'nonprofit': 'nonprofit charity social impact community development philanthropy fundraising volunteer management social services advocacy mission driven',
    
    'biotechnology': 'biotechnology life sciences genetics molecular biology drug discovery clinical research bioinformatics medical research pharmaceutical development',
    
    'insurance': 'insurance risk management underwriting claims processing actuarial analysis financial protection coverage assessment regulatory compliance',
    
    'architecture': 'architecture building design urban planning structural engineering construction management sustainable design architectural technology space planning',
    
    'environmental': 'environmental sustainability conservation ecology climate change renewable energy environmental consulting green technology waste management',
    
    'administrative_support': 'administrative support office management clerical services data entry scheduling coordination customer service executive assistance',
    
    'advertising_marketing': 'advertising marketing digital campaigns brand management creative services media planning performance marketing customer acquisition',
    
    'defense_aerospace': 'defense aerospace military technology aviation systems satellite technology security clearance government contracts space exploration',
    
    'facilities_management': 'facilities management building operations maintenance property services workplace solutions energy management space planning',
    
    'individual_family_services': 'social services family support counseling mental health community outreach welfare services case management crisis intervention',
    
    'internet_publishing': 'digital publishing online content web publishing content management SEO digital media editorial services online marketing',
    
    'market_research': 'market research consumer insights data analysis competitive intelligence trend analysis survey research business intelligence',
    
    'personal_care_manufacturing': 'personal care cosmetics beauty products consumer goods product formulation manufacturing retail distribution',
    
    'primary_secondary_education': 'K-12 education primary school secondary school teaching curriculum student development educational administration',
    
    'sporting_goods': 'sporting goods athletic equipment fitness products sports retail performance technology sports marketing active lifestyle',
    
    'venture_capital': 'venture capital private equity startup funding investment analysis deal sourcing portfolio management entrepreneurship',
    
    'quantum_computing': 'quantum computing quantum physics information technology advanced computing quantum algorithms scientific research',
    
    'sustainability_cleantech': 'sustainability clean technology environmental solutions renewable energy green innovation carbon reduction',
    
    'space_technology': 'space technology aerospace satellite systems space exploration orbital mechanics spacecraft design space commerce',
    
    'edge_computing': 'edge computing distributed systems real-time processing IoT connectivity low latency data processing network optimization',
    
    'extended_reality': 'extended reality virtual reality augmented reality mixed reality immersive technology spatial computing metaverse'
}
        
        # 排除规则
        mock_classifier.exclusion_rules = {
            'ui_ux_designer': {
                'exclude_skills': ['python', 'java', 'sql', 'backend_development'],
                'exclude_categories': ['programming_language', 'database']
            }
        }
        
        print("✅ 手动提取完成（使用基础数据集）")
        print("⚠️ 注意：这是简化的数据集，你可能需要手动补充完整数据")
        
    except Exception as e:
        print(f"❌ 手动提取也失败: {e}")
        return None
    
    return mock_classifier

def generate_role_config(classifier) -> Dict[str, Any]:
    """生成角色配置"""
    print("👥 生成角色配置...")
    
    roles = {}
    
    for role_name, mappings in classifier.explicit_role_mappings.items():
        semantic_desc = classifier.role_semantic_descriptions.get(
            role_name, 
            f"Professional role focused on {role_name.replace('_', ' ')} responsibilities and related tasks."
        )
        
        # 增强语义描述
        enhanced_descriptions = {
            'ui_ux_designer': """User interface and user experience design focusing on visual design, 
            interaction design, usability testing, wireframing, prototyping, design systems, 
            and user research. Creative problem-solving for digital products and user-centered design methodologies.""",
            
            'data_scientist': """Data analysis, statistical modeling, machine learning, and predictive analytics. 
            Extract insights from large datasets using programming, mathematics, and domain expertise. 
            Build predictive models, conduct A/B testing, and communicate findings to stakeholders.""",
            
            'frontend_web_developer': """Frontend web development focusing on user interface implementation, 
            responsive design, and interactive user experiences. JavaScript frameworks, HTML/CSS, 
            and modern web technologies for building engaging web applications.""",
            
            'backend_developer': """Backend software development focusing on server-side logic, APIs, 
            databases, and system architecture. Building scalable, secure, and performant backend systems 
            that power web and mobile applications."""
        }
        
        role_config = {
            'semantic_description': enhanced_descriptions.get(role_name, semantic_desc).strip(),
            'explicit_skills': mappings,
            'skill_type_preferences': {
                'tool': 0.7,
                'methodology': 0.6
            },
            'exclusion_rules': classifier.exclusion_rules.get(role_name, {
                'skills': [],
                'categories': []
            }),
            'alternative_names': [],
            'domain_keywords': [],
            'related_industries': []
        }
        
        roles[role_name] = role_config
        print(f"  ✅ {role_name}")
    
    return {'roles': roles}

def generate_industry_config(classifier) -> Dict[str, Any]:
    """生成行业配置"""
    print("🏭 生成行业配置...")
    
    industries = {}
    
    for industry_name, mappings in classifier.explicit_industry_mappings.items():
        semantic_desc = classifier.industry_semantic_descriptions.get(
            industry_name,
            f"Industry sector focused on {industry_name.replace('_', ' ')} domain and related business activities."
        )
        
        # 增强行业描述
        enhanced_descriptions = {
            'tech': """Technology sector focusing on software development, digital innovation, 
            artificial intelligence, cloud computing, and technology solutions. 
            Programming, system architecture, and cutting-edge technological advancement.""",
            
            'consulting': """Strategic business consulting, management consulting, digital transformation, 
            and organizational change. Problem-solving for complex business challenges, 
            process optimization, and strategic planning across various industries.""",
            
            'finance': """Financial services including banking, investment management, trading, 
            fintech, and wealth management. Risk assessment, financial analysis, 
            regulatory compliance, and financial technology innovation."""
        }
        
        industry_config = {
            'semantic_description': enhanced_descriptions.get(industry_name, semantic_desc).strip(),
            'explicit_skills': mappings,
            'domain_keywords': [],
            'related_roles': []
        }
        
        industries[industry_name] = industry_config
        print(f"  ✅ {industry_name}")
    
    return {'industries': industries}

def generate_engine_config(classifier) -> Dict[str, Any]:
    """生成引擎配置"""
    print("⚙️ 生成引擎配置...")
    
    config = {
        'semantic_engine': {
            'embedding_model': getattr(classifier.config, 'classification_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            'backup_model': getattr(classifier.config, 'backup_model', 'sentence-transformers/paraphrase-MiniLM-L6-v2'),
            
            'confidence_thresholds': {
                'high': getattr(classifier.config, 'high_confidence_threshold', 0.75),
                'medium': getattr(classifier.config, 'medium_confidence_threshold', 0.55),
                'low': getattr(classifier.config, 'low_confidence_threshold', 0.35)
            },
            
            'category_assignment_rules': {
                'essential_threshold': 0.85,
                'common_threshold': 0.75,
                'specialization_threshold': 0.65
            },
            
            'bonus_weights': {
                'skill_type_match': 0.15,
                'explicit_keyword_match': 0.25,
                'industry_role_alignment': 0.10
            }
        }
    }
    
    return config

def save_config_files(config_dir: Path, role_config: Dict, industry_config: Dict, engine_config: Dict):
    """保存配置文件"""
    print("💾 保存配置文件...")
    
    # 保存角色配置
    role_file = config_dir / "role_definitions.yaml"
    with open(role_file, 'w', encoding='utf-8') as f:
        yaml.dump(role_config, f, default_flow_style=False, allow_unicode=True, indent=2)
    print(f"  ✅ {role_file}")
    
    # 保存行业配置
    industry_file = config_dir / "industry_definitions.yaml"
    with open(industry_file, 'w', encoding='utf-8') as f:
        yaml.dump(industry_config, f, default_flow_style=False, allow_unicode=True, indent=2)
    print(f"  ✅ {industry_file}")
    
    # 保存引擎配置
    engine_file = config_dir / "semantic_engine_config.yaml"
    with open(engine_file, 'w', encoding='utf-8') as f:
        yaml.dump(engine_config, f, default_flow_style=False, allow_unicode=True, indent=2)
    print(f"  ✅ {engine_file}")

def main():
    """主迁移流程"""
    print("🚀 开始迁移 semantic_classifier.py 到配置文件...")
    print("📁 当前目录:", os.getcwd())
    print("=" * 60)
    
    try:
        # 1. 创建目录
        config_dir = create_directories()
        
        # 2. 提取数据
        classifier = extract_current_data()
        if classifier is None:
            print("❌ 无法提取数据，迁移终止")
            return False
        
        # 3. 生成配置
        role_config = generate_role_config(classifier)
        industry_config = generate_industry_config(classifier)
        engine_config = generate_engine_config(classifier)
        
        # 4. 保存文件
        save_config_files(config_dir, role_config, industry_config, engine_config)
        
        print("=" * 60)
        print("🎉 迁移完成！")
        print(f"📊 已迁移 {len(role_config['roles'])} 个角色和 {len(industry_config['industries'])} 个行业")
        print(f"📁 配置文件保存在: {config_dir}")
        print("\n📝 下一步:")
        print("1. 检查生成的config/目录下的YAML文件")
        print("2. 根据需要手动补充或修改配置")
        print("3. 测试新的配置系统")
        
        return True
        
    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)