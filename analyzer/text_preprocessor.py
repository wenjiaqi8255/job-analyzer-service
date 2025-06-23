import re
import pandas as pd
from extractor.industry_keyword_library import INDUSTRY_KEYWORD_LIBRARY
from .content_filter import ContentFilter


class TextPreprocessor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.stop_words = {
            # 德语停用词
            'und', 'der', 'die', 'das', 'für', 'von', 'mit', 'bei', 'den', 'dem', 'des',
            'ein', 'eine', 'einen', 'einer', 'sich', 'wir', 'sie', 'ihr', 'ich', 'du',
            'ist', 'sind', 'war', 'waren', 'haben', 'hat', 'wird', 'werden', 'kann',
            'soll', 'sollte', 'muss', 'auf', 'zu', 'nach', 'über', 'unter', 'durch',

            # 英语停用词
            'the', 'and', 'for', 'with', 'are', 'you', 'will', 'can', 'have', 'your',
            'our', 'this', 'that', 'work', 'working', 'job', 'position', 'role',
            'also', 'access', 'employees', 'company', 'team', 'opportunity',
            'during', 'right', 'therefore', 'should', 'needed', 'responsible',
            'something', 'experience', 'skills', 'requirements', 'candidates',

            # 招聘领域通用词
            'experience', 'skills', 'requirements', 'candidate', 'candidates',
            'position', 'role', 'job', 'work', 'working', 'opportunity',
            'team', 'company', 'employees', 'department', 'organization',
            'responsibilities', 'tasks', 'duties', 'qualifications',
            'background', 'knowledge', 'ability', 'abilities', 'capable',

            # 德语招聘通用词
            'stelle', 'position', 'arbeitsplatz', 'mitarbeiter', 'team',
            'unternehmen', 'firma', 'aufgaben', 'anforderungen', 'qualifikationen',
            'erfahrung', 'kenntnisse', 'fähigkeiten', 'verantwortung',
        }
        self.company_suffixes = {
            'gmbh', 'ag', 'kg', 'ohg', 'gbr', 'ug', 'eg', 'ev',
            'inc', 'corp', 'ltd', 'llc', 'co', 'company', 'group',
            'holding', 'ventures', 'capital', 'partners', 'solutions',
            'technologies', 'systems', 'services', 'consulting',
            'verlag', 'verlagsgruppe', 'media', 'publishing'
        }
        self.noise_patterns = [
            r'\w*_date_?\d*',  # publication_date_10
            r'job_id_?\d*',  # job_id_123
            r'location_\w+',  # location_munich
            r'\d{4,}',  # 长数字ID
            r'publication_date',  # 明确过滤
            r'legal_entity',  # 法律实体
            r'^.{1,2}$',  # 1-2字符的词
            r'^\d+$',  # 纯数字
            r'^[a-z]$',  # 单字母
            r'http[s]?://.*',  # URL
            r'.*@.*\..*',  # 邮箱
        ]
        self.meaningful_pos = {
            'NOUN', 'PROPN',  # 名词、专有名词
            'ADJ',  # 形容词
            'VERB',  # 动词
            'NUM'  # 数字
        }
        self.industry_cache = {}
        self.global_idf_cache = None  # This will be set from outside if needed
        self.content_filter = ContentFilter()

    def preprocess_job_title(self, job_title: str) -> str:
        """简单清理job title，去掉噪音词汇"""
        if not job_title or not job_title.strip():
            return ""

        title = job_title.strip()

        # 去掉资历级别词汇
        seniority_patterns = [
            r'\b(senior|sr\.?|lead|principal|staff|chief|head of|director of)\b',
            r'\b(junior|jr\.?|entry level|graduate|trainee|intern)\b',
            r'\b(mid|middle|intermediate|associate)\b'
        ]

        for pattern in seniority_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)

        # 去掉修饰词
        modifier_patterns = [
            r'\b(experienced|skilled|talented|expert|professional)\b',
            r'\b(full time|part time|freelance|contract|remote)\b',
            r'\s*\((?:m/f/d|w/m/d)\)|\b(?:m/f/d|w/m/d)\b'  # 德语性别标识
        ]

        for pattern in modifier_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)

        # 标准化分隔符和缩写
        title = re.sub(r'[-/&]', ' ', title)  # 统一分隔符
        title = re.sub(r'\bdev\b', 'developer', title, flags=re.IGNORECASE)
        title = re.sub(r'\beng\b', 'engineer', title, flags=re.IGNORECASE)
        title = re.sub(r'\bmgr\b', 'manager', title, flags=re.IGNORECASE)

        # 清理多余空格
        title = re.sub(r'\s+', ' ', title).strip()

        return title

    def classify_job_industry(self, company_name, job_title, description):
        # ... existing code ...
        cache_key = f"{company_name}_{job_title}"
        if cache_key in self.industry_cache:
            return self.industry_cache[cache_key]
        combined_text = f"{company_name} {job_title} {description}".lower()
        industry_keywords = INDUSTRY_KEYWORD_LIBRARY
        industry_scores = {}
        for industry, keywords in industry_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                industry_scores[industry] = score
        if industry_scores:
            classified_industry = max(industry_scores, key=industry_scores.get)
        else:
            classified_industry = 'general'
        self.industry_cache[cache_key] = classified_industry
        return classified_industry

    def is_generic_word(self, word, idf_threshold=2.5):
        if not self.global_idf_cache:
            return False
        idf_value = self.global_idf_cache.get(word.lower(), float('inf'))
        return idf_value < idf_threshold

    def extract_company_terms(self, company_name):
        if not company_name or pd.isna(company_name):
            return set()
        company_terms = set()
        company_name = str(company_name).lower()
        try:
            doc = self.nlp(company_name)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON']:
                    words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]{2,}\b', ent.text.lower())
                    company_terms.update(words)
        except:
            pass
        cleaned_name = re.sub(r'\([^)]*\)', '', company_name)
        cleaned_name = re.sub(r'[^\w\s\-]', ' ', cleaned_name)
        words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]{2,}\b', cleaned_name)
        company_terms.update(words)
        filtered_terms = set()
        for word in company_terms:
            if word not in self.company_suffixes:
                filtered_terms.add(word)
            else:
                filtered_terms.add(word)
        return filtered_terms

    def chunk_text(self, text: str) -> list[str]:
        """
        Splits a job description into semantic chunks (sentences or meaningful blocks).
        """
        if not text or pd.isna(text):
            return []

        # Replace multiple newlines/spaces, but keep single newlines to split on them later
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text).strip()
        
        final_chunks = []
        # Process each line separately
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            doc = self.nlp(line)
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if sent_text and len(sent_text.split()) >= 2:
                    final_chunks.append(sent_text)
        return final_chunks
    
    def chunk_and_filter_text(self, text: str) -> list[str]:
        """新增：chunk并过滤内容"""
        # Step 1: 正常chunking
        all_chunks = self.chunk_text(text)
        
        # Step 2: 内容过滤
        filtered_chunks = self.content_filter.filter_job_requirement_chunks(all_chunks)
        
        return filtered_chunks
    

    
    def advanced_text_preprocessing(self, text, company_terms_to_filter=None):
        """
        高级文本预处理，包括词形还原、过滤停用词、噪音词等
        company_terms_to_filter: 公司特定词汇
        bigram_contains_company: 二元组是否包含公司特定词汇
        unigrams: 一元组
        bigrams: 二元组
        valid_tokens: 有效token
        lemma: 词形还原
        """
        if not text or pd.isna(text):
            return [], []
        if company_terms_to_filter is None:
            company_terms_to_filter = set()
        text = str(text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'[^\w\s\-/]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        doc = self.nlp(text)
        unigrams = []
        bigrams = []
        valid_tokens = []
        for token in doc:
            if (token.is_punct or token.is_space or
                    token.text.lower() in self.stop_words or
                    self.is_noise_pattern(token.text) or
                    token.pos_ not in self.meaningful_pos):
                continue
            
            if token.text.lower() in company_terms_to_filter:
                continue

            lemma = token.lemma_.lower().strip()
            
            if len(lemma) < 3:
                continue

            valid_tokens.append(lemma)
            unigrams.append(lemma)
        for i in range(len(valid_tokens) - 1):
            bigram = f"{valid_tokens[i]} {valid_tokens[i+1]}"
            bigram_contains_company = any(
                company_term in bigram for company_term in company_terms_to_filter
            )
            if not bigram_contains_company:
                bigrams.append(bigram)
        return unigrams, bigrams

    def is_noise_pattern(self, word):
        for pattern in self.noise_patterns:
            if re.match(pattern, word, re.IGNORECASE):
                return True
        return False 