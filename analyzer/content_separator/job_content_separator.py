"""
Integration module that combines the SyntacticPatternClassifier with existing text preprocessing
to separate job content from company information in job postings.
"""

from typing import Dict, List, Any
from .syntactic_pattern_classifier import SyntacticPatternClassifier
from .content_filter import ContentFilter
import spacy
import re
import pandas as pd


class JobContentSeparator:
    """
    Service class that uses syntactic pattern analysis to separate job postings into
    actual job requirements/descriptions vs company introduction/information.
    """
    
    def __init__(self, nlp_model=None):
        if nlp_model is None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp_model
            
        self.classifier = SyntacticPatternClassifier(self.nlp)
        self.content_filter = ContentFilter()
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into sentences for processing.
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
    
    def separate_job_posting(self, job_description: str, confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Separate a job posting into company info and job content sections.
        
        Args:
            job_description: The full job posting text
            confidence_threshold: Minimum confidence to classify (default: 0.6)
            
        Returns:
            Dict containing separated content and analysis
        """
        if not job_description or not job_description.strip():
            return {
                'company_info_sentences': [],
                'job_content_sentences': [],
                'neutral_sentences': [],
                'company_info_text': '',
                'job_content_text': '',
                'analysis_summary': {
                    'total_sentences': 0,
                    'company_info_count': 0,
                    'job_content_count': 0,
                    'neutral_count': 0
                }
            }
        
        # Chunk text into sentences
        sentences = self._chunk_text(job_description)
        
        # Classify each sentence
        sentence_results = []
        company_sentences = []
        job_sentences = []
        neutral_sentences = []
        
        for sentence in sentences:
            result = self.classifier.classify_text(sentence)
            result['sentence'] = sentence
            sentence_results.append(result)
            
            # Categorize based on confidence threshold
            if result['confidence'] >= confidence_threshold:
                if result['type'] == 'company_info':
                    company_sentences.append(sentence)
                elif result['type'] == 'job_content':
                    job_sentences.append(sentence)
                else:
                    neutral_sentences.append(sentence)
            else:
                neutral_sentences.append(sentence)
        
        return {
            'company_info_sentences': company_sentences,
            'job_content_sentences': job_sentences,
            'neutral_sentences': neutral_sentences,
            'company_info_text': ' '.join(company_sentences),
            'job_content_text': ' '.join(job_sentences),
            'sentence_classifications': sentence_results,
            'analysis_summary': {
                'total_sentences': len(sentences),
                'company_info_count': len(company_sentences),
                'job_content_count': len(job_sentences),
                'neutral_count': len(neutral_sentences),
                'avg_confidence': sum(r['confidence'] for r in sentence_results) / len(sentence_results) if sentence_results else 0
            }
        }
    
    def extract_pure_job_requirements(self, job_description: str, 
                                    confidence_threshold: float = 0.7,
                                    apply_content_filter: bool = True) -> Dict[str, Any]:
        """
        Extract only the job requirements/descriptions, filtering out company info.
        Optionally applies additional content filtering.
        
        Args:
            job_description: The full job posting text
            confidence_threshold: Minimum confidence for job content classification
            apply_content_filter: Whether to apply additional content filtering
            
        Returns:
            Dict containing extracted job requirements and filtered content
        """
        separation_result = self.separate_job_posting(job_description, confidence_threshold)
        
        # Get job content sentences
        job_content_sentences = separation_result['job_content_sentences']
        
        # Optionally apply content filtering
        if apply_content_filter and job_content_sentences:
            filtered_sentences = self.content_filter.filter_job_requirement_chunks(job_content_sentences)
        else:
            filtered_sentences = job_content_sentences
        
        return {
            'original_text': job_description,
            'job_content_raw': ' '.join(job_content_sentences),
            'job_content_filtered': ' '.join(filtered_sentences) if filtered_sentences else '',
            'extracted_sentences': filtered_sentences,
            'removed_company_info': separation_result['company_info_text'],
            'filtering_stats': {
                'original_sentence_count': len(job_content_sentences),
                'filtered_sentence_count': len(filtered_sentences) if filtered_sentences else 0,
                'company_sentences_removed': len(separation_result['company_info_sentences']),
                'neutral_sentences': len(separation_result['neutral_sentences'])
            }
        }
    
    def analyze_job_posting_structure(self, job_description: str) -> Dict[str, Any]:
        """
        Analyze the overall structure of a job posting to understand content distribution.
        
        Args:
            job_description: The job posting text to analyze
            
        Returns:
            Detailed analysis of the job posting structure
        """
        separation_result = self.separate_job_posting(job_description)
        sentence_classifications = separation_result['sentence_classifications']
        
        # Calculate position-based statistics
        total_sentences = len(sentence_classifications)
        company_positions = []
        job_positions = []
        
        for i, result in enumerate(sentence_classifications):
            if result['type'] == 'company_info':
                company_positions.append(i)
            elif result['type'] == 'job_content':
                job_positions.append(i)
        
        # Analyze patterns
        company_at_start = len([pos for pos in company_positions if pos < total_sentences * 0.3]) if company_positions else 0
        company_at_end = len([pos for pos in company_positions if pos > total_sentences * 0.7]) if company_positions else 0
        job_in_middle = len([pos for pos in job_positions if 0.3 <= pos/total_sentences <= 0.7]) if job_positions else 0
        
        return {
            'structure_analysis': {
                'total_sentences': total_sentences,
                'company_info_positions': company_positions,
                'job_content_positions': job_positions,
                'company_at_beginning': company_at_start,
                'company_at_end': company_at_end,
                'job_content_in_middle': job_in_middle,
                'typical_structure': self._identify_typical_structure(company_positions, job_positions, total_sentences)
            },
            'content_distribution': {
                'company_info_ratio': len(company_positions) / total_sentences if total_sentences > 0 else 0,
                'job_content_ratio': len(job_positions) / total_sentences if total_sentences > 0 else 0,
                'neutral_ratio': (total_sentences - len(company_positions) - len(job_positions)) / total_sentences if total_sentences > 0 else 0
            },
            'quality_indicators': {
                'has_clear_job_content': len(job_positions) >= 3,
                'excessive_company_info': len(company_positions) > len(job_positions),
                'well_structured': self._is_well_structured(company_positions, job_positions, total_sentences),
                'avg_classification_confidence': separation_result['analysis_summary']['avg_confidence']
            }
        }
    
    def _identify_typical_structure(self, company_positions: List[int], job_positions: List[int], total: int) -> str:
        """Identify the typical structure pattern of the job posting."""
        if not company_positions and not job_positions:
            return "unclear"
        
        if not company_positions:
            return "job_only"
        
        if not job_positions:
            return "company_only"
        
        company_start = min(company_positions) if company_positions else total
        job_start = min(job_positions) if job_positions else total
        
        if company_start < job_start:
            return "company_first_then_job"
        elif job_start < company_start:
            return "job_first_then_company"
        else:
            return "mixed_throughout"
    
    def _is_well_structured(self, company_positions: List[int], job_positions: List[int], total: int) -> bool:
        """Determine if the job posting has a clear, well-organized structure."""
        if total < 3:
            return False
        
        # Check if job content ratio is reasonable (at least 40% of content)
        job_ratio = len(job_positions) / total if total > 0 else 0
        if job_ratio < 0.4:
            return False
        
        # Check if there's not excessive company info (more than 50% is excessive)
        company_ratio = len(company_positions) / total if total > 0 else 0
        if company_ratio > 0.5:
            return False
        
        return True
    
    def process_job_listing_batch(self, job_listings: List[Dict[str, str]], 
                                job_desc_field: str = 'description') -> List[Dict[str, Any]]:
        """
        Process multiple job listings to extract pure job requirements.
        
        Args:
            job_listings: List of job listing dictionaries
            job_desc_field: Field name containing job description
            
        Returns:
            List of processed job listings with separated content
        """
        results = []
        
        for listing in job_listings:
            if job_desc_field not in listing:
                continue
                
            job_description = listing[job_desc_field]
            
            # Extract pure job requirements
            extraction_result = self.extract_pure_job_requirements(job_description)
            
            # Add to result
            processed_listing = listing.copy()
            processed_listing.update({
                'original_description': job_description,
                'job_requirements_only': extraction_result['job_content_filtered'],
                'company_info_removed': extraction_result['removed_company_info'],
                'extraction_stats': extraction_result['filtering_stats']
            })
            
            results.append(processed_listing)
        
        return results