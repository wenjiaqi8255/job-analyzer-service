import re
import pandas as pd
from extractor.industry_keyword_library import INDUSTRY_KEYWORD_LIBRARY
from .content_filter import ContentFilter
from typing import List


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

    def preprocess_text(self, text: str) -> str:
        """
        A general-purpose text preprocessing function.
        Lemmatizes, removes stopwords and noise, and returns a clean string.
        """
        if not text or pd.isna(text):
            return ""

        doc = self.nlp(text)
        lemmas = [token.lemma_.lower() for token in doc if self._is_meaningful_token(token)]
        
        return " ".join(lemmas)

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
    

    
    def process_text_for_embedding(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[List[str]]:
        """
        Processes text into overlapping chunks of lemmas, suitable for embedding models.
        This is designed to capture local context by creating sliding windows of tokens.
        """
        if not text or pd.isna(text):
            return []

        # Clean and lemmatize
        lemmas = [token.lemma_.lower() for token in self.nlp(text) if self._is_meaningful_token(token)]
        
        # Create overlapping chunks
        chunks = []
        for i in range(0, len(lemmas), chunk_size - overlap):
            chunk = lemmas[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
                
        return chunks

    def is_noise_pattern(self, word):
        for pattern in self.noise_patterns:
            if re.match(pattern, word, re.IGNORECASE):
                return True
        return False

    def improved_chunking_for_anomaly_detection(self, text: str) -> list[str]:
        """
        专为异常检测优化的chunking方法, 结合了多种策略.
        """
        # Step 1: 先对整个文本进行初步过滤，去除明显无关的部分
        # Note: filter_job_requirement_chunks 期望一个chunks列表
        # 我们将整个文本视为一个chunk进行过滤
        filtered_text_list = self.content_filter.filter_job_requirement_chunks([text])
        if not filtered_text_list:
            return []
        filtered_text = filtered_text_list[0]

        # Step 2: 应用改进的chunking策略
        # 基本的句子级别的chunk
        primary_chunks = self._sentence_level_chunking(filtered_text)
        
        # 跨句子的chunk
        cross_sentence_chunks = self._cross_sentence_chunking(filtered_text)

        # Step 3: 合并并去重
        all_chunks = primary_chunks + cross_sentence_chunks
        unique_chunks = list(dict.fromkeys(all_chunks))  # 保留顺序的去重

        # Step 4: 质量过滤
        final_chunks = [chunk for chunk in unique_chunks if self._is_meaningful_chunk(chunk)]

        return final_chunks

    def _sentence_level_chunking(self, text: str) -> list[str]:
        """
        基于句子进行切分 (类似原有的 chunk_text)
        """
        if not text or pd.isna(text):
            return []

        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text).strip()
        
        chunks = []
        doc = self.nlp(text)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            # 使用更有意义的判断
            if len(sent_text.split()) >= 3:
                chunks.append(sent_text)
        return chunks

    def _cross_sentence_chunking(self, text: str) -> list[str]:
        """
        创建跨越相邻句子的chunk，以捕捉上下文关系.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        chunks = []
        for i in range(len(sentences) - 1):
            sent1 = sentences[i]
            sent2 = sentences[i+1]
            
            # 检查两个句子是否可能相关
            if self._are_related_requirements(sent1, sent2):
                combined = f"{sent1} {sent2}"
                # 避免组合过长的chunk
                if len(combined.split()) <= 25:
                    chunks.append(combined)
        return chunks

    def _are_related_requirements(self, sent1: str, sent2: str) -> bool:
        """
        判断两个相邻的句子是否是相关的技能/要求描述.
        """
        # 关键词可以更丰富一些
        tech_keywords = [
            'experience', 'proficiency', 'knowledge', 'skills', 'required', 
            'preferred', 'familiar', 'background', 'understanding', 'degree',
            'bachelor', 'master', 'phd', 'qualification', 'bonus'
        ]
        
        s1_lower = sent1.lower()
        s2_lower = sent2.lower()

        s1_has_tech = any(keyword in s1_lower for keyword in tech_keywords)
        s2_has_tech = any(keyword in s2_lower for keyword in tech_keywords)
        
        # 规则1: 如果两个句子都包含技术/要求关键词，则可能相关
        if s1_has_tech and s2_has_tech:
            return True
            
        # 规则2: 如果第一句以冒号结尾，通常是列表的开始
        if sent1.endswith(':'):
            return True
        
        # 规则3: 如果两个句子都比较短，且缺乏动词，可能是技能列表的一部分
        s1_words = len(sent1.split())
        s2_words = len(sent2.split())
        if s1_words < 10 and s2_words < 10:
            # 这是一个简化的检查，可以做得更复杂
            # 比如检查是否存在动词
            doc1 = self.nlp(sent1)
            doc2 = self.nlp(sent2)
            s1_has_verb = any(token.pos_ == 'VERB' for token in doc1)
            s2_has_verb = any(token.pos_ == 'VERB' for token in doc2)
            if not s1_has_verb and not s2_has_verb:
                return True

        return False

    def _is_meaningful_chunk(self, chunk: str) -> bool:
        """
        判断一个chunk是否有足够的分析价值.
        """
        words = chunk.split()
        
        # 长度过滤
        if len(words) < 3 or len(words) > 30:
            return False
            
        # 技术背景词检查
        tech_context_words = [
            'experience', 'skills', 'knowledge', 'proficiency', 'familiar', 'expert', 
            'required', 'preferred', 'background', 'understanding', 'tools'
        ]
        has_tech_context = any(word in chunk.lower() for word in tech_context_words)
        
        # 实质性内容检查 (避免都是停用词或通用词)
        # 使用正则表达式匹配包含字母的单词
        meaningful_words = len(re.findall(r'\b[a-zA-Z]{3,}\b', chunk))

        # 如果包含技术背景词，或者有足够多的实质性单词，则认为是有意义的
        return has_tech_context or meaningful_words >= 3 

    def _is_meaningful_token(self, token):
        """Checks if a token is worth keeping for analysis."""
        if (token.is_stop or
                token.is_punct or
                token.is_space or
                token.lemma_.lower() in self.stop_words or
                self.is_noise_pattern(token.lemma_.lower()) or
                token.pos_ not in self.meaningful_pos):
            return False
        return True 