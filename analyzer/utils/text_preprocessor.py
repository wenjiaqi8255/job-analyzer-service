import logging
import re
import pandas as pd
from extractor.industry_keyword_library import INDUSTRY_KEYWORD_LIBRARY
from ..content_separator.content_filter import ContentFilter
from typing import List

logger = logging.getLogger(__name__)


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

        logger.debug(f"Original job title: '{job_title}'")
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

        logger.debug(f"Preprocessed job title: '{title}'")
        return title

    def preprocess_text(self, text: str) -> str:
        """
        A general-purpose text preprocessing function.
        Lemmatizes, removes stopwords and noise, and returns a clean string.
        """
        if not text or pd.isna(text):
            return ""

        logger.debug(f"Original text for preprocessing: '{text[:100]}...'")
        doc = self.nlp(text)
        lemmas = [token.lemma_.lower() for token in doc if self._is_meaningful_token(token)]
        
        processed_text = " ".join(lemmas)
        logger.debug(f"Preprocessed text (lemmas): '{processed_text[:100]}...'")
        return processed_text

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

        logger.debug(f"Text for chunking: '{text[:100]}...'")
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
        
        logger.debug(f"Found {len(final_chunks)} chunks.")
        return final_chunks
    
    def chunk_and_filter_text(self, text: str) -> list[str]:
        """新增：chunk并过滤内容"""
        # Step 1: 正常chunking
        all_chunks = self.chunk_text(text)
        logger.debug(f"Generated {len(all_chunks)} chunks before filtering.")
        
        # Step 2: 内容过滤
        filtered_chunks = self.content_filter.filter_job_requirement_chunks(all_chunks)
        logger.debug(f"Found {len(filtered_chunks)} chunks after filtering.")
        
        return filtered_chunks
    

    
    def process_text_for_embedding(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[List[str]]:
        """
        Processes text into overlapping chunks of lemmas, suitable for embedding models.
        This is designed to capture local context by creating sliding windows of tokens.
        """
        if not text or pd.isna(text):
            return []

        logger.debug(f"Processing text for embedding: '{text[:100]}...'")
        # Clean and lemmatize
        lemmas = [token.lemma_.lower() for token in self.nlp(text) if self._is_meaningful_token(token)]
        logger.debug(f"Generated {len(lemmas)} lemmas.")
        
        # Create overlapping chunks
        chunks = []
        for i in range(0, len(lemmas), chunk_size - overlap):
            chunk = lemmas[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        
        logger.debug(f"Created {len(chunks)} chunks for embedding.")
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
        logger.debug(f"Starting improved chunking for anomaly detection on text: '{text[:100]}...'")
        # Step 1: 先对整个文本进行初步过滤，去除明显无关的部分
        # Note: filter_job_requirement_chunks 期望一个chunks列表
        # 我们将整个文本视为一个chunk进行过滤
        filtered_text_list = self.content_filter.filter_job_requirement_chunks([text])
        if not filtered_text_list:
            logger.debug("Text completely filtered out, returning empty list.")
            return []
        filtered_text = filtered_text_list[0]
        logger.debug(f"Text after initial filtering: '{filtered_text[:100]}...'")

        # Step 2: 应用改进的chunking策略
        # 基本的句子级别的chunk
        primary_chunks = self._sentence_level_chunking(filtered_text)
        logger.debug(f"Generated {len(primary_chunks)} primary chunks.")
        
        # 跨句子的chunk
        cross_sentence_chunks = self._cross_sentence_chunking(filtered_text)
        logger.debug(f"Generated {len(cross_sentence_chunks)} cross-sentence chunks.")

        # Step 3: 合并并去重
        all_chunks = list(dict.fromkeys(primary_chunks + cross_sentence_chunks))

        # Step 4: 过滤掉无意义的chunk
        final_chunks = [chunk for chunk in all_chunks if self._is_meaningful_chunk(chunk)]
        
        logger.debug(f"Total {len(final_chunks)} unique meaningful chunks after merging.")

        return final_chunks

    def _sentence_level_chunking(self, text: str) -> list[str]:
        """
        Performs sentence-level chunking of the text.
        """
        if not text:
            return []
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) > 1]

    def _cross_sentence_chunking(self, text: str) -> list[str]:
        """
        Creates chunks by combining related adjacent sentences.
        """
        sents = self._sentence_level_chunking(text)
        if len(sents) < 2:
            return sents

        chunks = []
        i = 0
        while i < len(sents):
            current_sent = sents[i]
            if i + 1 < len(sents):
                next_sent = sents[i+1]
                # Heuristic to combine related sentences
                if self._are_related_requirements(current_sent, next_sent):
                    chunks.append(current_sent + " " + next_sent)
                    i += 2  # Skip next sentence
                    continue
            chunks.append(current_sent)
            i += 1
        return chunks

    def _are_related_requirements(self, sent1: str, sent2: str) -> bool:
        """
        A heuristic to determine if two sentences describe related aspects of a job requirement.
        Example: "你必须有5年Java经验" 和 "并且熟悉Spring框架"
        """
        sent1_lower = sent1.lower()
        sent2_lower = sent2.lower()

        # 检查第二句是否以连词开头 (e.g., and, also, plus)
        coordinating_conjunctions = [
            'and', 'also', 'plus', 'in addition', 'additionally',
            'und', 'auch', 'sowie', 'zudem', 'darüber hinaus', 'des Weiteren'
        ]
        if any(sent2_lower.startswith(conj) for conj in coordinating_conjunctions):
            return True

        # 检查句子结构相似性 (e.g., both start with "experience with...")
        # (这是一个简化的例子，可以用更复杂的NLP方法)
        tokens1 = [token.lemma_ for token in self.nlp(sent1_lower) if not token.is_stop and not token.is_punct]
        tokens2 = [token.lemma_ for token in self.nlp(sent2_lower) if not token.is_stop and not token.is_punct]
        
        # 如果第二句很短，并且和第一句有重叠，可能是一个补充说明
        if len(tokens2) < 5 and len(set(tokens1) & set(tokens2)) > 0:
            return True

        # 检查是否有共同的实体或技术词汇 (这里可以引入关键词列表)
        # 例子: "Java", "Python", "experience", "Kenntnisse"
        common_tech_keywords = {'java', 'python', 'c++', 'javascript', 'react', 'vue', 'angular', 'sql', 'nosql', 'docker', 'kubernetes', 'aws', 'azure', 'gcp'}
        if len(set(tokens1) & common_tech_keywords) > 0 and len(set(tokens2) & common_tech_keywords) > 0:
            return True

        return False

    def _is_meaningful_chunk(self, chunk: str) -> bool:
        """
        Determines if a chunk of text is meaningful for analysis, not just boilerplate.
        """
        chunk_lower = chunk.lower()

        # 过滤掉过于通用的"我们提供"类型的句子
        boilerplate_starters = [
            'we offer', 'wir bieten', 'what we offer', 'unser angebot', 'your benefits',
            'what you can expect', 'was wir dir bieten', 'was dich erwartet',
            'we are looking for', 'wir suchen', 'you are', 'du bist',
            'about us', 'über uns', 'who we are'
        ]
        if any(chunk_lower.startswith(starter) for starter in boilerplate_starters):
            return False

        # 过滤掉联系信息
        if 'contact' in chunk_lower or 'ansprechpartner' in chunk_lower or 'bewerbung' in chunk_lower:
            return False

        # 过滤掉只有一个词的chunk
        if len(chunk.split()) <= 2:
            return False

        # 至少要包含一个有意义的词性
        doc = self.nlp(chunk)
        has_meaningful_pos = any(token.pos_ in self.meaningful_pos for token in doc)
        if not has_meaningful_pos:
            return False

        return True

    def _is_meaningful_token(self, token):
        """
        Checks if a token is worth keeping for analysis.
        """
        # Not a stop word
        if token.is_stop:
            return False
        # Not punctuation
        if token.is_punct:
            return False
        # Not a space
        if token.is_space:
            return False
        # Has a meaningful part-of-speech tag
        if token.pos_ not in self.meaningful_pos:
            return False
        # Not a noise pattern and not a generic word
        lemma = token.lemma_.lower()
        if self.is_noise_pattern(lemma) or lemma in self.stop_words:
            return False
            
        return True 