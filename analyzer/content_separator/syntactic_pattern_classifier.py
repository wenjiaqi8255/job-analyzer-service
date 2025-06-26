import spacy
from typing import Dict, List, Any


class SyntacticPatternClassifier:
    """
    Classifier to distinguish between "actual job descriptions" and "company introductions" 
    in job postings using syntactic patterns rather than semantic analysis.
    """
    
    def __init__(self, nlp_model=None):
        if nlp_model is None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp_model
        
        # Company introduction patterns (focus on "we", "our", company-centric language)
        self.company_patterns = [
            # We + verb patterns
            {"pattern": [{"LOWER": "we"}, {"POS": "AUX"}, {"POS": "VERB"}]},
            {"pattern": [{"LOWER": "we"}, {"POS": "VERB"}]},
            # Our + noun patterns
            {"pattern": [{"LOWER": "our"}, {"POS": "NOUN"}]},
            {"pattern": [{"LOWER": "our"}, {"POS": "ADJ"}, {"POS": "NOUN"}]},
            # Company + verb patterns
            {"pattern": [{"LOWER": "company"}, {"POS": "AUX"}, {"POS": "VERB"}]},
            {"pattern": [{"LOWER": "company"}, {"POS": "VERB"}]},
            # Passive voice + company founding/establishment
            {"pattern": [{"POS": "VERB", "DEP": "auxpass"}, {"LOWER": {"IN": ["founded", "established", "created"]}}]},
            # Organization patterns
            {"pattern": [{"LOWER": "organization"}, {"POS": "VERB"}]},
            {"pattern": [{"LOWER": "team"}, {"POS": "VERB"}]}
        ]
        
        # Job description patterns (focus on "you", requirements, responsibilities)
        self.job_patterns = [
            # You + verb patterns
            {"pattern": [{"LOWER": "you"}, {"POS": "AUX"}, {"POS": "VERB"}]},
            {"pattern": [{"LOWER": "you"}, {"POS": "VERB"}]},
            # Candidate patterns
            {"pattern": [{"LOWER": "candidate"}, {"POS": "AUX"}]},
            {"pattern": [{"LOWER": "candidate"}, {"POS": "VERB"}]},
            # Imperative patterns (verb as root without explicit subject)
            {"pattern": [{"POS": "VERB", "DEP": "ROOT"}]},
            # Requirement expressions
            {"pattern": [{"LOWER": {"IN": ["must", "required", "need", "should"]}}, {"POS": "VERB"}]},
            {"pattern": [{"LOWER": {"IN": ["must", "required", "need", "should"]}}, {"POS": "AUX"}]},
            # Responsibility patterns
            {"pattern": [{"LOWER": "responsible"}, {"LOWER": "for"}]},
            {"pattern": [{"LOWER": "responsibilities"}, {"LOWER": "include"}]}
        ]
        
        # Company-specific keywords
        self.company_keywords = {
            'company_descriptors': ['founded', 'established', 'headquartered', 'specializes', 'leading', 'pioneer'],
            'company_pronouns': ['we', 'our', 'us'],
            'company_entities': ['company', 'organization', 'firm', 'corporation', 'enterprise', 'startup']
        }
        
        # Job-specific keywords
        self.job_keywords = {
            'candidate_refs': ['you', 'candidate', 'applicant', 'individual'],
            'requirements': ['must', 'required', 'should', 'need', 'essential', 'mandatory'],
            'responsibilities': ['responsible', 'duties', 'tasks', 'role', 'position'],
            'skills': ['experience', 'skills', 'knowledge', 'proficiency', 'expertise', 'background']
        }
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify a text segment as either company info or job content.
        
        Args:
            text: The text to classify
            
        Returns:
            Dict containing classification result and analysis details
        """
        if not text or not text.strip():
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'company_score': 0.0,
                'job_score': 0.0,
                'syntax_analysis': {}
            }
        
        doc = self.nlp(text)
        
        # Analyze syntactic patterns
        company_score = self._analyze_company_syntax(doc)
        job_score = self._analyze_job_syntax(doc)
        
        # Determine classification
        if company_score > job_score:
            classification = 'company_info'
            confidence = company_score / (company_score + job_score) if (company_score + job_score) > 0 else 0.5
        elif job_score > company_score:
            classification = 'job_content'
            confidence = job_score / (company_score + job_score) if (company_score + job_score) > 0 else 0.5
        else:
            classification = 'neutral'
            confidence = 0.5
        
        return {
            'type': classification,
            'confidence': confidence,
            'company_score': company_score,
            'job_score': job_score,
            'syntax_analysis': self._get_syntax_features(doc)
        }
    
    def classify_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        Classify each sentence in the text individually.
        
        Args:
            text: The text to process
            
        Returns:
            List of classification results for each sentence
        """
        if not text or not text.strip():
            return []
        
        doc = self.nlp(text)
        results = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                result = self.classify_text(sent_text)
                result['sentence'] = sent_text
                results.append(result)
        
        return results
    
    def _analyze_company_syntax(self, doc) -> float:
        """Analyze syntactic features indicating company introduction."""
        score = 0.0
        
        for token in doc:
            # Check for company pronouns as subjects
            if token.dep_ == "nsubj" and token.lower_ in ["we", "company", "organization", "firm"]:
                score += 0.2
            
            # Check for "we" as subject with verbs
            if token.lower_ == "we" and token.dep_ == "nsubj":
                score += 0.15
                
            # Check for possessive "our"
            if token.lower_ == "our" and token.pos_ == "PRON":
                score += 0.1
                
            # Check for company-related verbs
            if token.lower_ in ["founded", "established", "headquartered", "specializes"] and token.pos_ == "VERB":
                score += 0.3
                
            # Check for passive voice with company founding
            if token.dep_ == "auxpass" and any(child.lower_ in ["founded", "established", "created"] 
                                            for child in token.head.children):
                score += 0.3
        
        # Boost score for company-centric sentence structures
        subjects = [token.text.lower() for token in doc if token.dep_ == "nsubj"]
        if any(subj in ["we", "our company", "the company"] for subj in subjects):
            score += 0.2
            
        return min(score, 1.0)  # Cap at 1.0
    
    def _analyze_job_syntax(self, doc) -> float:
        """Analyze syntactic features indicating job description."""
        score = 0.0
        
        for token in doc:
            # Check for "you" as subject
            if token.dep_ == "nsubj" and token.lower_ in ["you", "candidate", "applicant"]:
                score += 0.4
            
            # Check for imperative sentences (verb as root without explicit subject)
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                has_explicit_subject = any(child.dep_ == "nsubj" for child in token.children)
                if not has_explicit_subject:
                    score += 0.2  # Likely imperative
            
            # Check for modal verbs indicating requirements
            if token.lower_ in ["must", "should", "will", "need", "preferred", "nice to have", "advantageous", "beneficial", "plus"] and token.pos_ in ["AUX", "VERB"]:
                score += 0.3
                
            # Check for requirement keywords
            if token.lower_ in ["required", "essential", "mandatory", "necessary"]:
                score += 0.2
                
            # Check for responsibility indicators
            if token.lower_ in ["responsible", "duties", "tasks", "role"]:
                score += 0.2
        
        # Boost score for job-centric sentence structures
        root_verbs = [token.text.lower() for token in doc if token.dep_ == "ROOT" and token.pos_ == "VERB"]
        if any(verb in ["manage", "develop", "create", "implement", "design", "maintain"] for verb in root_verbs):
            score += 0.2
            
        return min(score, 1.0)  # Cap at 1.0
    
    def _get_syntax_features(self, doc) -> Dict[str, Any]:
        """Extract syntactic features for analysis and debugging."""
        features = {
            'subjects': [token.text for token in doc if token.dep_ == "nsubj"],
            'root_verbs': [token.text for token in doc if token.dep_ == "ROOT" and token.pos_ == "VERB"],
            'modals': [token.text for token in doc if token.pos_ == "AUX"],
            'pronouns': [token.text for token in doc if token.pos_ == "PRON"],
            'has_imperative': self._has_imperative_structure(doc),
            'has_passive': self._has_passive_structure(doc),
            'sentence_length': len([token for token in doc if not token.is_punct and not token.is_space])
        }
        return features
    
    def _has_imperative_structure(self, doc) -> bool:
        """Check if the sentence has imperative structure."""
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                has_explicit_subject = any(child.dep_ == "nsubj" for child in token.children)
                if not has_explicit_subject:
                    return True
        return False
    
    def _has_passive_structure(self, doc) -> bool:
        """Check if the sentence has passive voice structure."""
        return any(token.dep_ == "auxpass" for token in doc)
    
    def batch_classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple texts in batch.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of classification results
        """
        results = []
        for text in texts:
            result = self.classify_text(text)
            results.append(result)
        return results
    
    def get_classification_summary(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get a summary of classifications for a list of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Summary statistics of the classification results
        """
        results = self.batch_classify(texts)
        
        company_count = sum(1 for r in results if r['type'] == 'company_info')
        job_count = sum(1 for r in results if r['type'] == 'job_content')
        neutral_count = sum(1 for r in results if r['type'] == 'neutral')
        
        avg_company_score = sum(r['company_score'] for r in results) / len(results) if results else 0
        avg_job_score = sum(r['job_score'] for r in results) / len(results) if results else 0
        
        return {
            'total_texts': len(texts),
            'company_info_count': company_count,
            'job_content_count': job_count,
            'neutral_count': neutral_count,
            'avg_company_score': avg_company_score,
            'avg_job_score': avg_job_score,
            'classification_ratio': {
                'company_info': company_count / len(texts) if texts else 0,
                'job_content': job_count / len(texts) if texts else 0,
                'neutral': neutral_count / len(texts) if texts else 0
            }
        }