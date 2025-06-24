import unittest
from unittest.mock import Mock, MagicMock
from analyzer.classifier.classification_pipeline import ClassificationPipeline
from analyzer.config import ModelThresholds

class TestClassificationPipeline(unittest.TestCase):

    def setUp(self):
        """Set up mocked components for each test."""
        self.mock_embedding_model = Mock()
        self.mock_semantic_baselines = {'role': {'some_role': {}}}
        self.mock_text_preprocessor = Mock()
        self.mock_thresholds = ModelThresholds()

        # Mock the RoleClassifier which is instantiated inside the pipeline
        self.mock_role_classifier = Mock()
        self.mock_role_classifier.classify_job_role.return_value = "software_engineer"

        # We need to mock the RoleClassifier's constructor
        self.patcher = unittest.mock.patch('analyzer.classification_pipeline.RoleClassifier', return_value=self.mock_role_classifier)
        self.MockRoleClassifier = self.patcher.start()
        self.addCleanup(self.patcher.stop)

        self.pipeline = ClassificationPipeline(
            embedding_model=self.mock_embedding_model,
            semantic_baselines=self.mock_semantic_baselines,
            text_preprocessor=self.mock_text_preprocessor,
            thresholds=self.mock_thresholds
        )

    def test_run_classification(self):
        """Test the full run method of the classification pipeline."""
        job_data = {
            'job_title': 'Senior Software Engineer',
            'effective_description': 'We are looking for a dev to write code.',
            'industry': 'Technology'
        }

        expected_result = {
            "role": "software_engineer",
            "industry": "Technology"
        }

        # Run the pipeline
        result = self.pipeline.run(job_data)

        # Assert that the role classifier was called correctly
        self.mock_role_classifier.classify_job_role.assert_called_once_with(
            job_title='Senior Software Engineer',
            job_description='We are looking for a dev to write code.'
        )

        # Assert that the final output is correct
        self.assertEqual(result, expected_result)

    def test_run_with_missing_data(self):
        """Test how the pipeline handles missing job title, description, and industry."""
        job_data = {}

        # The role classifier should still return the default mock value
        self.mock_role_classifier.classify_job_role.return_value = "data_scientist"

        expected_result = {
            "role": "data_scientist",
            "industry": "general"  # Should default to 'general'
        }

        result = self.pipeline.run(job_data)

        self.mock_role_classifier.classify_job_role.assert_called_once_with(
            job_title='',
            job_description=''
        )
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main() 