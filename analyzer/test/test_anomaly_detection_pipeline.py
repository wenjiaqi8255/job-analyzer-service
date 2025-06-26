import unittest
from unittest.mock import Mock, patch
from analyzer.anomaly_detector.anomaly_detection_pipeline import AnomalyDetectionPipeline
from analyzer.config import ModelThresholds, PipelineConfig, CacheConfig

class TestAnomalyDetectionPipeline(unittest.TestCase):

    def setUp(self):
        """Set up mocked components for each test."""
        self.mock_embedding_model = Mock()
        self.mock_semantic_baselines = {'role': {}, 'industry': {}, 'global': {}}
        self.mock_text_preprocessor = Mock()
        self.mock_nlp_model = Mock()
        self.mock_thresholds = ModelThresholds()
        self.mock_pipeline_config = PipelineConfig()
        self.mock_cache_config = CacheConfig()

        # Mock the SemanticAnomalyDetector which is instantiated inside the pipeline
        self.mock_anomaly_detector = Mock()
        self.mock_anomaly_detector.detect_anomalies.return_value = [{"type": "Cross-Role", "chunk": "test chunk"}]
        
        # Patch the constructor of SemanticAnomalyDetector
        self.patcher = patch('analyzer.anomaly_detection_pipeline.SemanticAnomalyDetector', return_value=self.mock_anomaly_detector)
        self.MockSemanticAnomalyDetector = self.patcher.start()
        self.addCleanup(self.patcher.stop)

        self.pipeline = AnomalyDetectionPipeline(
            embedding_model=self.mock_embedding_model,
            semantic_baselines=self.mock_semantic_baselines,
            text_preprocessor=self.mock_text_preprocessor,
            nlp_model=self.mock_nlp_model,
            thresholds=self.mock_thresholds,
            pipeline_config=self.mock_pipeline_config,
            cache_config=self.mock_cache_config
        )

    def test_run_anomaly_detection(self):
        """Test the full run method of the anomaly detection pipeline."""
        job_data = {
            'effective_description': 'Some job description text here.'
        }
        classification_result = {
            'role': 'software_engineer',
            'industry': 'Technology'
        }

        expected_anomalies = [{"type": "Cross-Role", "chunk": "test chunk"}]

        # Run the pipeline
        anomalies = self.pipeline.run(job_data, classification_result)

        # Assert that the anomaly detector was called correctly
        self.mock_anomaly_detector.detect_anomalies.assert_called_once_with(
            target_job='Some job description text here.',
            role_baseline_name='software_engineer',
            industry_baseline_name='Technology'
        )

        # Assert that the final output is correct
        self.assertEqual(anomalies, expected_anomalies)

    def test_run_with_missing_classification(self):
        """Test how the pipeline handles missing role and industry."""
        job_data = {
            'effective_description': 'Another job description.'
        }
        classification_result = {
            'role': None,
            'industry': None
        }
        
        # Run the pipeline
        self.pipeline.run(job_data, classification_result)

        # Assert that the anomaly detector was still called with the None values
        self.mock_anomaly_detector.detect_anomalies.assert_called_once_with(
            target_job='Another job description.',
            role_baseline_name=None,
            industry_baseline_name=None
        )

if __name__ == '__main__':
    unittest.main() 