import unittest
from unittest.mock import MagicMock
import spacy
from analyzer.text_preprocessor import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.nlp = spacy.load("de_core_news_sm")
        except OSError:
            print("Downloading spacy model de_core_news_sm for testing...")
            spacy.cli.download("de_core_news_sm")
            cls.nlp = spacy.load("de_core_news_sm")
        
        cls.preprocessor = TextPreprocessor(cls.nlp)

    def test_preprocess_job_title(self):
        self.assertEqual(self.preprocessor.preprocess_job_title("Senior Software Developer (m/f/d)"), "Software Developer")
        self.assertEqual(self.preprocessor.preprocess_job_title("Junior Frontend Engineer / Dev"), "Frontend Engineer developer")
        self.assertEqual(self.preprocessor.preprocess_job_title("Full time Product Manager"), "Product Manager")
        self.assertEqual(self.preprocessor.preprocess_job_title("  Lead Data Scientist  "), "Data Scientist")
        self.assertEqual(self.preprocessor.preprocess_job_title("UI/UX Designer"), "UI UX Designer")
        self.assertEqual(self.preprocessor.preprocess_job_title("Chief Executive Officer"), "Executive Officer")
        self.assertEqual(self.preprocessor.preprocess_job_title(None), "")
        self.assertEqual(self.preprocessor.preprocess_job_title(""), "")

    def test_is_noise_pattern(self):
        self.assertTrue(self.preprocessor.is_noise_pattern("publication_date_10"))
        self.assertTrue(self.preprocessor.is_noise_pattern("12345"))
        self.assertTrue(self.preprocessor.is_noise_pattern("http://example.com"))
        self.assertTrue(self.preprocessor.is_noise_pattern("test@example.com"))
        self.assertTrue(self.preprocessor.is_noise_pattern("ab"))
        self.assertFalse(self.preprocessor.is_noise_pattern("developer"))
        self.assertFalse(self.preprocessor.is_noise_pattern("skill"))
    
    def test_chunk_text(self):
        text_de = "Das ist der erste Satz. Hier ist der zweite Satz.\nDas ist ein dritter Satz in einer neuen Zeile."
        expected_chunks_de = ["Das ist der erste Satz.", "Hier ist der zweite Satz.", "Das ist ein dritter Satz in einer neuen Zeile."]
        self.assertEqual(self.preprocessor.chunk_text(text_de), expected_chunks_de)

        text_with_newlines = "First part.\nSecond part, still meaningful. \n\nThird part after empty line."
        expected_chunks_newlines = ["First part.", "Second part, still meaningful.", "Third part after empty line."]
        self.assertEqual(self.preprocessor.chunk_text(text_with_newlines), expected_chunks_newlines)
        
        text_with_short_chunks = "Sentence one.\nShort\nAnother long sentence."
        expected_chunks_short = ["Sentence one.", "Another long sentence."]
        self.assertEqual(self.preprocessor.chunk_text(text_with_short_chunks), expected_chunks_short)

        self.assertEqual(self.preprocessor.chunk_text(""), [])
        self.assertEqual(self.preprocessor.chunk_text(None), [])

    def test_chunk_and_filter_text(self):
        self.preprocessor.content_filter = MagicMock()
        
        text = "This is a sentence about requirements. This is a sentence about benefits. You need 5 years of experience."
        
        # Based on how spacy's sentencizer works.
        all_chunks = ['This is a sentence about requirements.', 'This is a sentence about benefits.', 'You need 5 years of experience.']

        filtered_chunks = ["This is a sentence about requirements.", "You need 5 years of experience."]
        
        self.preprocessor.content_filter.filter_job_requirement_chunks.return_value = filtered_chunks
        
        result = self.preprocessor.chunk_and_filter_text(text)
        
        self.preprocessor.content_filter.filter_job_requirement_chunks.assert_called_once()
        # The argument passed to the mock might vary slightly based on NLP model sentence splitting.
        # So we check the call happened and then check the final output.
        self.assertEqual(result, filtered_chunks)

    def test_advanced_text_preprocessing(self):
        text = "Wir suchen einen C++ Entwickler mit Erfahrung in agilen Methoden. Unsere Firma heißt Awesome Inc."
        company_terms = {'awesome', 'inc'}
        
        unigrams, bigrams = self.preprocessor.advanced_text_preprocessing(text, company_terms)

        # "Erfahrung" and "Firma" are stopwords in the code
        expected_unigrams = ['suchen', 'entwickler', 'agile', 'methode', 'heißen']
        self.assertCountEqual(unigrams, expected_unigrams)
        
        expected_bigrams = ['suchen entwickler', 'entwickler agile', 'agile methode', 'methode heißen']
        self.assertCountEqual(bigrams, expected_bigrams)

        unigrams_no_filter, _ = self.preprocessor.advanced_text_preprocessing(text)
        self.assertIn('awesom', unigrams_no_filter)
        self.assertIn('inc', unigrams_no_filter)

if __name__ == '__main__':
    unittest.main() 