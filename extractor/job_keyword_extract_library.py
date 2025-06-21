# # JobbAI 德国求职市场专业词库
# # 用于TF-IDF异常词检测和职位匹配算法

# JOB_KEYWORD_LIBRARY = {
#     # ==================== 技术类词汇 ====================
#     'technical': {
#         # 编程语言
#         'programming_languages': [
#             'python', 'java', 'javascript', 'typescript', 'go', 'rust', 'c++', 'c#', 'php', 'ruby',
#             'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'dart', 'perl',
#             'shell', 'bash', 'powershell', 'assembly', 'cobol', 'fortran'
#         ],
        
#         # 前端技术
#         'frontend': [
#             'react', 'vue', 'angular', 'svelte', 'jquery', 'bootstrap', 'sass', 'less', 'webpack',
#             'vite', 'parcel', 'gulp', 'npm', 'yarn', 'pnpm', 'jsx', 'tsx', 'styled-components',
#             'mui', 'antd', 'tailwindcss', 'nextjs', 'nuxtjs', 'gatsby', 'redux', 'mobx', 'zustand'
#         ],
        
#         # 后端技术
#         'backend': [
#             'django', 'flask', 'fastapi', 'spring', 'spring boot', 'express', 'nodejs', 'nest',
#             'laravel', 'symfony', 'rails', 'sinatra', 'asp.net', 'gin', 'fiber', 'echo',
#             'rest', 'restful', 'graphql', 'grpc', 'soap', 'api', 'microservices', 'serverless'
#         ],
        
#         # 数据库
#         'database': [
#             'mysql', 'postgresql', 'sqlite', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
#             'dynamodb', 'firebase', 'supabase', 'prisma', 'sequelize', 'mongoose', 'typeorm',
#             'knex', 'drizzle', 'nosql', 'acid', 'oltp', 'olap', 'etl', 'data warehouse',
#             'snowflake', 'bigquery', 'redshift', 'clickhouse'
#         ],
        
#         # 云计算与DevOps
#         'cloud_devops': [
#             'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'vagrant',
#             'jenkins', 'github actions', 'gitlab ci', 'circle ci', 'travis ci', 'ci/cd',
#             'helm', 'istio', 'prometheus', 'grafana', 'elk', 'datadog', 'newrelic',
#             'nginx', 'apache', 'cloudflare', 'cdn', 'load balancer', 'auto scaling'
#         ],
        
#         # 数据科学与AI
#         'data_ai': [
#             'machine learning', 'deep learning', 'artificial intelligence', 'neural networks',
#             'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
#             'jupyter', 'anaconda', 'mlflow', 'kubeflow', 'airflow', 'spark', 'hadoop',
#             'nlp', 'computer vision', 'reinforcement learning', 'transformers', 'bert', 'gpt',
#             'llm', 'generative ai', 'stable diffusion', 'hugging face', 'openai', 'anthropic'
#         ],
        
#         # 移动开发
#         'mobile': [
#             'ios', 'android', 'swift', 'objective-c', 'kotlin', 'java android', 'react native',
#             'flutter', 'xamarin', 'ionic', 'cordova', 'phonegap', 'native script',
#             'xcode', 'android studio', 'firebase', 'push notifications', 'in-app purchases'
#         ],
        
#         # 测试与质量保证
#         'testing_qa': [
#             'unit testing', 'integration testing', 'e2e testing', 'test automation',
#             'jest', 'mocha', 'jasmine', 'cypress', 'selenium', 'playwright', 'testcafe',
#             'junit', 'pytest', 'rspec', 'tdd', 'bdd', 'code coverage', 'sonarqube'
#         ],
        
#         # 版本控制与协作
#         'version_control': [
#             'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial', 'perforce',
#             'pull request', 'merge request', 'code review', 'branching', 'gitflow'
#         ],
        
#         # 安全
#         'security': [
#             'cybersecurity', 'penetration testing', 'vulnerability assessment', 'oauth', 'jwt',
#             'ssl', 'tls', 'https', 'encryption', 'authentication', 'authorization', 'rbac',
#             'sso', 'ldap', 'active directory', 'firewall', 'vpn', 'zero trust', 'devsecops'
#         ]
#     },
    
#     # ==================== 行业领域词汇 ====================
#     'domain': {
#         # 金融科技
#         'fintech': [
#             'blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'defi', 'nft', 'web3',
#             'smart contracts', 'trading algorithms', 'robo advisor', 'regtech', 'insurtech',
#             'payment processing', 'digital banking', 'open banking', 'psd2', 'kyc', 'aml'
#         ],
        
#         # 传统金融
#         'finance': [
#             'investment banking', 'asset management', 'portfolio management', 'risk management',
#             'derivatives', 'equity', 'bonds', 'fixed income', 'forex', 'commodities',
#             'basel', 'mifid', 'solvency', 'ifrs', 'gaap', 'var', 'stress testing',
#             'credit risk', 'market risk', 'operational risk', 'compliance', 'audit'
#         ],
        
#         # ESG与可持续发展
#         'esg_sustainability': [
#             'esg', 'sustainability', 'carbon footprint', 'net zero', 'renewable energy',
#             'circular economy', 'green finance', 'impact investing', 'social impact',
#             'governance', 'stakeholder capitalism', 'sustainable development goals', 'sdg',
#             'climate risk', 'transition risk', 'physical risk', 'tcfd', 'sfdr'
#         ],
        
#         # 法律与合规
#         'legal_compliance': [
#             'gdpr', 'data protection', 'privacy', 'litigation', 'regulatory', 'compliance',
#             'contract law', 'corporate law', 'intellectual property', 'ip', 'patent',
#             'trademark', 'copyright', 'due diligence', 'merger', 'acquisition', 'm&a',
#             'antitrust', 'competition law', 'employment law', 'labor law'
#         ],
        
#         # 咨询
#         'consulting': [
#             'strategy consulting', 'management consulting', 'digital transformation',
#             'business process optimization', 'change management', 'organizational design',
#             'market research', 'competitive analysis', 'case study', 'framework',
#             'mckinsey', 'bcg', 'bain', 'pwc', 'deloitte', 'ey', 'kpmg'
#         ],
        
#         # 医疗保健
#         'healthcare_life_sciences': [
#             'pharmaceutical', 'biotech', 'medical device', 'clinical trials', 'drug discovery',
#             'fda', 'ema', 'gcp', 'gmp', 'regulatory affairs', 'pharmacovigilance',
#             'medical writing', 'clinical research', 'bioinformatics', 'genomics', 'proteomics',
#             'digital health', 'telemedicine', 'health tech', 'medtech'
#         ],
        
#         # 汽车工业
#         'automotive': [
#             'automotive', 'electric vehicle', 'ev', 'autonomous driving', 'adas',
#             'mobility', 'connected car', 'car sharing', 'ride sharing', 'micromobility',
#             'battery technology', 'charging infrastructure', 'automotive software',
#             'cariad', 'volkswagen digital', 'bmw', 'mercedes', 'audi', 'porsche'
#         ],
        
#         # 制造业与工业4.0
#         'manufacturing_industry40': [
#             'industry 4.0', 'smart manufacturing', 'iot', 'industrial iot', 'iiot',
#             'automation', 'robotics', 'cobots', 'digital twin', 'predictive maintenance',
#             'supply chain', 'lean manufacturing', 'six sigma', 'kaizen', 'erp',
#             'mes', 'scada', 'plc', 'hmi', 'opc ua'
#         ],
        
#         # 电商与零售
#         'ecommerce_retail': [
#             'e-commerce', 'online marketplace', 'dropshipping', 'omnichannel',
#             'customer experience', 'personalization', 'recommendation engine',
#             'inventory management', 'warehouse management', 'logistics', 'fulfillment',
#             'payment gateway', 'checkout optimization', 'conversion rate optimization'
#         ],
        
#         # 媒体与广告
#         'media_advertising': [
#             'digital marketing', 'performance marketing', 'programmatic advertising',
#             'ad tech', 'mar tech', 'seo', 'sem', 'social media marketing', 'content marketing',
#             'influencer marketing', 'affiliate marketing', 'email marketing', 'crm',
#             'customer segmentation', 'attribution modeling', 'marketing automation'
#         ],
        
#         # 游戏
#         'gaming': [
#             'game development', 'unity', 'unreal engine', 'c# unity', 'game design',
#             'mobile gaming', 'casual games', 'hyper casual', 'mmo', 'moba',
#             'game monetization', 'in-app purchases', 'freemium', 'gaas',
#             'user acquisition', 'retention', 'ltv', 'arpu'
#         ],
        
#         # 房地产科技
#         'proptech': [
#             'proptech', 'real estate', 'property management', 'facility management',
#             'smart building', 'building automation', 'energy management',
#             'rental platform', 'property valuation', 'mortgage tech', 'construction tech'
#         ]
#     },
    
#     # ==================== 德国特色词汇 ====================
#     'german_specific': {
#         # 德国技术生态
#         'german_tech': [
#             'sap', 'siemens', 'volkswagen digital', 'cariad', 'deutsche telekom',
#             'rocket internet', 'zalando', 'delivery hero', 'about you', 'trivago',
#             'n26', 'trade republic', 'celonis', 'personio', 'contentful'
#         ],
        
#         # 德国法律法规
#         'german_legal': [
#             'bafin', 'bundesdatenschutzgesetz', 'bdsg', 'betriebsrat', 'works council',
#             'mitbestimmung', 'tarifvertrag', 'collective bargaining', 'kurzarbeit',
#             'elterngeld', 'parental leave', 'bundesagentur für arbeit'
#         ],
        
#         # 德国商业文化
#         'german_business': [
#             'mittelstand', 'konzern', 'gmbh', 'ag', 'dax', 'mdax', 'tecdax',
#             'startup ecosystem', 'berlin startup', 'munich startup', 'hamburg startup',
#             'cologne startup', 'frankfurt fintech', 'industry 4.0', 'energiewende'
#         ],
        
#         # 德国地区与城市
#         'german_locations': [
#             'berlin', 'munich', 'münchen', 'hamburg', 'cologne', 'köln', 'frankfurt',
#             'stuttgart', 'düsseldorf', 'dortmund', 'essen', 'leipzig', 'dresden',
#             'hanover', 'hannover', 'nuremberg', 'nürnberg', 'ruhr area', 'bavaria',
#             'bayern', 'baden-württemberg', 'north rhine-westphalia', 'nrw'
#         ]
#     },
    
#     # ==================== 通用商业工具 ====================
#     'business_tools': [
#         # Office套件
#         'microsoft office', 'excel', 'powerpoint', 'word', 'outlook', 'teams',
#         'google workspace', 'google sheets', 'google docs', 'google slides',
#         'office 365', 'sharepoint', 'onedrive', 'slack', 'discord', 'zoom',
        
#         # ERP与业务系统
#         'sap', 'oracle', 'netsuite', 'dynamics 365', 'workday', 'servicenow',
#         'atlassian', 'jira', 'confluence', 'trello', 'asana', 'monday.com',
#         'notion', 'airtable', 'smartsheet',
        
#         # CRM与销售
#         'salesforce', 'hubspot', 'pipedrive', 'zoho', 'freshworks', 'intercom',
        
#         # 数据分析工具
#         'tableau', 'power bi', 'qlik', 'looker', 'databricks', 'alteryx',
#         'r studio', 'spss', 'stata', 'sas', 'matlab', 'mathematica',
        
#         # 设计工具
#         'figma', 'sketch', 'adobe creative suite', 'photoshop', 'illustrator',
#         'indesign', 'after effects', 'premiere pro', 'canva', 'invision'
#     ],
    
#     # ==================== 软技能 ====================
#     'soft_skills': [
#         # 领导力
#         'leadership', 'team leadership', 'people management', 'coaching', 'mentoring',
#         'strategic thinking', 'vision', 'decision making', 'delegation',
        
#         # 沟通协作
#         'communication', 'presentation skills', 'public speaking', 'negotiation',
#         'collaboration', 'teamwork', 'cross-functional', 'stakeholder management',
#         'client relations', 'customer service',
        
#         # 分析思维
#         'analytical thinking', 'problem solving', 'critical thinking', 'data driven',
#         'quantitative analysis', 'qualitative analysis', 'research', 'insights',
        
#         # 项目管理
#         'project management', 'agile', 'scrum', 'kanban', 'waterfall', 'pmp',
#         'risk management', 'budget management', 'timeline management',
        
#         # 创新创业
#         'innovation', 'creativity', 'entrepreneurship', 'startup mindset',
#         'growth hacking', 'product thinking', 'user centric', 'design thinking',
        
#         # 适应性
#         'adaptability', 'flexibility', 'resilience', 'change management',
#         'continuous learning', 'growth mindset', 'curiosity'
#     ],
    
#     # ==================== 职位要求与条件 ====================
#     'job_requirements': {
#         # 工作模式
#         'work_mode': [
#             'remote', 'hybrid', 'on-site', 'flexible', 'home office', 'co-working',
#             'distributed team', 'async', 'synchronous', '4 day week', 'part time',
#             'full time', 'contract', 'freelance', 'permanent', 'temporary'
#         ],
        
#         # 经验水平
#         'experience_level': [
#             'entry level', 'junior', 'mid level', 'senior', 'lead', 'principal',
#             'staff', 'architect', 'director', 'vp', 'c-level', 'executive',
#             'intern', 'working student', 'graduate', 'trainee', 'apprentice'
#         ],
        
#         # 语言要求
#         'language_requirements': [
#             'german', 'deutsch', 'english', 'bilingual', 'native speaker',
#             'fluent', 'conversational', 'basic', 'b1', 'b2', 'c1', 'c2',
#             'french', 'spanish', 'italian', 'mandarin', 'japanese'
#         ],
        
#         # 学历要求
#         'education': [
#             'bachelor', 'master', 'phd', 'mba', 'university', 'fachhochschule',
#             'tu', 'technical university', 'computer science', 'engineering',
#             'business administration', 'economics', 'mathematics', 'physics',
#             'bootcamp', 'certification', 'self taught'
#         ],
        
#         # 福利待遇
#         'benefits': [
#             'equity', 'stock options', 'bonus', 'pension', 'health insurance',
#             'dental', 'gym membership', 'learning budget', 'conference budget',
#             'sabbatical', 'unlimited vacation', 'flexible hours', 'company car',
#             'bike leasing', 'meal vouchers', 'childcare', 'relocation support'
#         ]
#     },
    
#     # ==================== 新兴技术趋势 ====================
#     'emerging_trends': [
#         # AI与ML新趋势
#         'chatgpt', 'large language models', 'prompt engineering', 'rag',
#         'vector databases', 'embeddings', 'fine tuning', 'llmops',
#         'multimodal ai', 'computer vision', 'edge ai', 'federated learning',
        
#         # Web3与区块链
#         'web3', 'defi', 'dao', 'smart contracts', 'solidity', 'ethereum',
#         'polygon', 'avalanche', 'cardano', 'polkadot', 'chainlink',
        
#         # 量子计算
#         'quantum computing', 'quantum algorithms', 'qiskit', 'cirq',
        
#         # AR/VR/元宇宙
#         'augmented reality', 'virtual reality', 'mixed reality', 'metaverse',
#         'unity ar', 'arkit', 'arcore', 'oculus', 'hololens',
        
#         # 边缘计算与IoT
#         'edge computing', 'fog computing', 'internet of things', 'iot',
#         'smart sensors', 'digital twin', 'real time analytics',
        
#         # 低代码/无代码
#         'low code', 'no code', 'citizen developer', 'workflow automation',
#         'zapier', 'power automate', 'retool', 'bubble', 'webflow'
#     ]
# }

# # ==================== 词库使用工具函数 ====================

# def get_all_keywords():
#     """获取所有关键词的扁平化列表"""
#     all_keywords = []
    
#     def extract_keywords(data):
#         if isinstance(data, list):
#             all_keywords.extend(data)
#         elif isinstance(data, dict):
#             for value in data.values():
#                 extract_keywords(value)
    
#     extract_keywords(JOB_KEYWORD_LIBRARY)
#     return list(set(all_keywords))  # 去重

# def get_keywords_by_category(category):
#     """根据类别获取关键词"""
#     return JOB_KEYWORD_LIBRARY.get(category, [])

# def search_keywords(query):
#     """搜索包含特定字符串的关键词"""
#     query = query.lower()
#     all_keywords = get_all_keywords()
#     return [kw for kw in all_keywords if query in kw.lower()]

# def get_keyword_count():
#     """获取词库统计信息"""
#     all_keywords = get_all_keywords()
#     return {
#         'total_keywords': len(all_keywords),
#         'categories': len(JOB_KEYWORD_LIBRARY),
#         'technical_keywords': len(get_all_keywords_from_category('technical')),
#         'domain_keywords': len(get_all_keywords_from_category('domain')),
#         'german_specific': len(get_all_keywords_from_category('german_specific'))
#     }

# def get_all_keywords_from_category(category):
#     """从特定大类中获取所有关键词"""
#     keywords = []
#     category_data = JOB_KEYWORD_LIBRARY.get(category, {})
    
#     def extract_from_dict(data):
#         if isinstance(data, list):
#             keywords.extend(data)
#         elif isinstance(data, dict):
#             for value in data.values():
#                 extract_from_dict(value)
    
#     extract_from_dict(category_data)
#     return keywords

# # ==================== 使用示例 ====================

# if __name__ == "__main__":
#     # 获取统计信息
#     stats = get_keyword_count()
#     print(f"词库统计: {stats}")
    
#     # 搜索特定关键词
#     ai_keywords = search_keywords("ai")
#     print(f"包含'ai'的关键词: {ai_keywords[:10]}")  # 只显示前10个
    
#     # 获取特定类别的关键词
#     fintech_keywords = get_keywords_by_category('domain')['fintech']
#     print(f"金融科技关键词: {fintech_keywords[:5]}")  # 只显示前5个