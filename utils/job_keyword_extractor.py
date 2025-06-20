import re
from collections import Counter
from utils.job_keyword_extract_library import JOB_KEYWORD_LIBRARY

class KeywordExtractor:
    def __init__(self):
        self.keyword_map = self._build_keyword_map()

    def _build_keyword_map(self):
        """
        Builds a reverse map from a keyword to its category and sub-category.
        Example: {'python': ('technical', 'programming_languages'), 'react': ('technical', 'frontend')}
        """
        keyword_map = {}
        for category, sub_categories in JOB_KEYWORD_LIBRARY.items():
            if isinstance(sub_categories, dict):
                for sub_category, keywords in sub_categories.items():
                    for keyword in keywords:
                        keyword_map[keyword.lower()] = (category, sub_category)
            elif isinstance(sub_categories, list):
                # Handle flat lists like 'business_tools' and 'soft_skills'
                for keyword in sub_categories:
                    keyword_map[keyword.lower()] = (category, category) # Use main category as sub-category
        return keyword_map

    def _normalize_text(self, text):
        """Basic text normalization."""
        text = text.lower()
        # Pad special characters to ensure they are tokenized, e.g., "C++" -> " c++ "
        text = re.sub(r'([#+])', r' \1 ', text) 
        # Remove other punctuation
        text = re.sub(r'[^\w\s#+]', ' ', text)
        # Remove extra whitespace
        text = ' ' + ' '.join(text.split()) + ' '
        return text

    def extract_keywords(self, text):
        """
        Extracts and categorizes keywords from a given text.
        Returns a dictionary with counts for each keyword type.
        """
        if not text or not isinstance(text, str):
            return {}

        normalized_text = self._normalize_text(text)
        
        extracted_keywords = {
            'technical': Counter(),
            'domain': Counter(),
            'german_specific': Counter(),
            'business_tools': Counter(),
            'soft_skills': Counter(),
            'job_requirements': Counter(),
            'emerging_trends': Counter()
        }

        # Use a simple text search which is fast and effective for keyword libraries.
        # This avoids complex tokenization issues with multi-word keywords.
        for keyword, (category, sub_category) in self.keyword_map.items():
            # Use word boundaries for single-word keywords to avoid matching substrings
            # For multi-word keywords or keywords with special characters, a simple 'in' check is better.
            if ' ' in keyword or '#' in keyword or '+' in keyword:
                if keyword in normalized_text:
                     count = normalized_text.count(keyword)
                     extracted_keywords[category][keyword] += count
            else:
                # Pad with spaces for exact word matching
                pattern = f' {keyword} '
                if pattern in normalized_text:
                    count = normalized_text.count(pattern)
                    extracted_keywords[category][keyword] += count

        # Convert counters to simple dicts for JSON serialization
        result = {key: dict(value) for key, value in extracted_keywords.items() if value}
        return result

# Example usage
if __name__ == '__main__':
    extractor = KeywordExtractor()
    sample_description = """
    We are looking for a Senior Python Developer with experience in Django and React.
    You will be working with AWS, Docker, and Kubernetes.
    Experience in the Fintech industry and knowledge of ESG is a big plus.
    Must be fluent in German (C1). This is a remote role. We use Jira and Slack.
    """
    
    keywords = extractor.extract_keywords(sample_description)
    
    import json
    print(json.dumps(keywords, indent=2))
    
    # Expected output:
    # {
    #   "technical": {
    #     "python": 1,
    #     "django": 1,
    #     "react": 1,
    #     "aws": 1,
    #     "docker": 1,
    #     "kubernetes": 1
    #   },
    #   "domain": {
    #     "fintech": 1,
    #     "esg": 1
    #   },
    #   "job_requirements": {
    #     "german": 1,
    #     "c1": 1,
    #     "remote": 1
    #   },
    #   "business_tools": {
    #      "jira": 1,
    #      "slack": 1
    #   }
    # } 