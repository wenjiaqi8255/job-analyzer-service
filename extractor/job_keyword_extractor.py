# from keybert import KeyBERT


# class KeywordExtractor:
#     """
#     Extracts keywords from text using the KeyBERT model.
#     This implementation uses a pre-trained multilingual model to find relevant
#     keywords and keyphrases, replacing the old static dictionary-based approach.
#     """
#     def __init__(self):
#         """
#         Initializes the KeywordExtractor with a multilingual KeyBERT model.
#         """
#         self.kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

#     def extract_keywords(self, text: str) -> list[tuple[str, float]]:
#         """
#         Extracts keywords and their similarity scores from a given text.
        
#         Args:
#             text: The input string to extract keywords from.

#         Returns:
#             A list of (keyword, score) tuples. Returns an empty list if input is invalid.
#         """
#         if not text or not isinstance(text, str):
#             return []

#         # Parameters are based on the refactoring plan in `doc/keyword_extract_v2.md`
#         # - `keyphrase_ngram_range`: To capture single keywords up to 3-word phrases.
#         # - `stop_words`: Set to None to handle multiple languages (German/English).
#         # - `top_n`: Extracts the top 30 most relevant keywords.
#         # - `use_mmr`: Use Maximal Marginal Relevance to diversify results.
#         # - `diversity`: A higher value (0.7) is chosen to encourage discovery of
#         #                more unique or "abnormal" terms, aligning with project goals.
#         keywords_with_scores = self.kw_model.extract_keywords(
#             text,
#             keyphrase_ngram_range=(1, 2),
#             stop_words='english',
#             top_n=30,
#             use_mmr=True,
#             diversity=0.7
#         )
        
#         # As per the updated plan, return the full result with scores.
#         return keywords_with_scores

# # Example usage
# if __name__ == '__main__':
#     extractor = KeywordExtractor()
#     sample_description = """
#     We are looking for a Senior Python Developer with experience in Django and React.
#     You will be working with AWS, Docker, and Kubernetes.
#     Experience in the Fintech industry and knowledge of ESG is a big plus.
#     Must be fluent in German (C1). This is a remote role. We use Jira and Slack.
#     Ein deutsches Beispiel: Wir suchen einen erfahrenen Softwareentwickler für die Arbeit mit KI-Modellen.
#     """
    
#     keywords = extractor.extract_keywords(sample_description)
    
#     import json
#     print("Extracted Keywords with Scores:")
#     print(json.dumps(keywords, indent=2, ensure_ascii=False))
    
#     # Expected output is a list of [keyword, score] pairs, for example:
#     # [
#     #   [
#     #     "erfahrenen softwareentwickler für",
#     #     0.65
#     #   ],
#     #   [
#     #     "senior python developer",
#     #     0.62
#     #   ],
#     #   ...
#     # ] 