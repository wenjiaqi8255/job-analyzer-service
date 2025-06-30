import unittest
from unittest.mock import Mock, patch
from analyzer.baseline_classifier.analysis_pipeline import AnalysisPipeline
from analyzer.config import AppConfig

class TestAnalysisPipeline(unittest.TestCase):

    def setUp(self):
        """Set up mocked components for each test."""
        self.mock_embedding_model = Mock()
        self.mock_semantic_baselines = {
            'role': {'software_engineer': {}, 'data_scientist': {}},
            'industry': {'tech': {}, 'finance': {}}
        }
        self.mock_text_preprocessor = Mock()
        self.mock_config = AppConfig()

        # Mock the analyzers
        self.mock_role_analyzer = Mock()
        self.mock_industry_analyzer = Mock()

        # We need to mock the analyzers' constructors
        self.role_patcher = patch('analyzer.baseline_classifier.analysis_pipeline.RoleSimilarityAnalyzer', return_value=self.mock_role_analyzer)
        self.industry_patcher = patch('analyzer.baseline_classifier.analysis_pipeline.IndustrySimilarityAnalyzer', return_value=self.mock_industry_analyzer)
        
        self.MockRoleAnalyzer = self.role_patcher.start()
        self.MockIndustryAnalyzer = self.industry_patcher.start()
        
        self.addCleanup(self.role_patcher.stop)
        self.addCleanup(self.industry_patcher.stop)

        self.pipeline = AnalysisPipeline(
            embedding_model=self.mock_embedding_model,
            semantic_baselines=self.mock_semantic_baselines,
            text_preprocessor=self.mock_text_preprocessor,
            config=self.mock_config
        )

    def test_analyze_successful(self):
        """Test the full analyze method of the analysis pipeline."""
        job_data = {
            'job_title': 'Senior Software Engineer',
            'effective_description': 'We are looking for a dev to write code.',
            'company_name': 'Tech Corp',
            'industry': 'Technology',
            'company_description': 'A company that does tech.'
        }

        # Configure mock return values
        self.mock_role_analyzer.calculate_similarities.return_value = {
            "software_engineer": 0.9,
            "data_scientist": 0.4
        }
        self.mock_industry_analyzer.calculate_similarities.return_value = {
            "tech": 0.85,
            "finance": 0.3
        }

        # Run the pipeline
        result = self.pipeline.analyze(job_data)

        # Assert that the analyzers were called correctly
        self.mock_role_analyzer.calculate_similarities.assert_called_once_with(
            'Senior Software Engineer', 'We are looking for a dev to write code.'
        )
        self.mock_industry_analyzer.calculate_similarities.assert_called_once_with(
            'Technology', 'Tech Corp', 'A company that does tech.'
        )

        # Assert that the final output is correct
        self.assertEqual(result['role'], "software_engineer")
        self.assertEqual(result['industry'], "tech")
        self.assertIn('role_similarity_analysis', result)
        self.assertIn('industry_similarity_analysis', result)
        self.assertEqual(result['role_similarity_analysis']['software_engineer'], 0.9)
        self.assertEqual(result['industry_similarity_analysis']['tech'], 0.85)

    def test_analyze_with_low_scores(self):
        """Test that primary role/industry fall back correctly with low scores."""
        job_data = {}

        # Configure mock return values
        self.mock_role_analyzer.calculate_similarities.return_value = {"software_engineer": 0.4}
        self.mock_industry_analyzer.calculate_similarities.return_value = {"tech": 0.3}
        
        # Set thresholds high enough to fail
        self.pipeline.config.model_thresholds.role_similarity_threshold = 0.5
        self.pipeline.config.model_thresholds.industry_similarity_threshold = 0.5

        result = self.pipeline.analyze(job_data)

        self.assertEqual(result['role'], "general")
        self.assertEqual(result['industry'], "Unknown")
        self.assertEqual(result['role_similarity_analysis']['software_engineer'], 0.4)

if __name__ == '__main__':
    unittest.main() 