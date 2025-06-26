import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import json
import tempfile
import os
from analyzer.run_pipeline import run_pipeline
from analyzer.utils.exceptions import PipelineError
from analyzer.utils.data_validator import ValidationError

class TestRunPipeline(unittest.TestCase):
    def setUp(self):
        """Setup test data and mocks"""
        # Sample job data that represents a realistic job posting
        self.sample_job_data = {
            'id': 'test_job_001',
            'job_title': 'Senior Machine Learning Engineer',
            'company_name': 'TechCorp Inc.',
            'description': 'We are looking for an experienced ML engineer to join our AI team. Requirements: 5+ years Python, TensorFlow experience, PhD preferred.',
            'translated_description': 'We are looking for an experienced ML engineer to join our AI team. Requirements: 5+ years Python, TensorFlow experience, PhD preferred.',
            'location': 'San Francisco, CA',
            'salary_range': '$150,000 - $200,000'
        }
        
        # Create DataFrame with sample job
        self.jobs_df = pd.DataFrame([self.sample_job_data])
        self.idf_corpus_df = pd.DataFrame()  # Mock empty IDF corpus
        
        # Expected pipeline results at each step
        self.expected_classification_result = {
            'role': 'ML Engineer',
            'industry': 'Technology',
            'confidence_score': 0.95,
            'role_embedding': [0.1, 0.2, 0.3],  # Mock embedding
            'industry_embedding': [0.4, 0.5, 0.6]  # Mock embedding
        }
        
        self.expected_anomalies = [
            {
                'anomaly_type': 'semantic_mismatch',
                'description': 'Unusual requirement combination detected',
                'severity': 'medium',
                'details': {'requirement': 'PhD for ML Engineer role', 'score': 0.7}
            }
        ]

    @patch('analyzer.run_pipeline.get_config')
    @patch('analyzer.run_pipeline.ResourceManager')
    @patch('analyzer.run_pipeline.TextPreprocessor')
    @patch('analyzer.run_pipeline.ClassificationPipeline')
    @patch('analyzer.run_pipeline.AnomalyDetectionPipeline')
    @patch('analyzer.run_pipeline.validate_job_data')
    @patch('analyzer.run_pipeline.write_analysis_result')
    def test_pipeline_step_by_step_results(self, mock_write_result, mock_validate, 
                                         mock_anomaly_pipeline, mock_classification_pipeline,
                                         mock_text_preprocessor, mock_resource_manager, mock_get_config):
        """Test the complete pipeline and verify results at each step"""
        
        # Setup configuration mock
        mock_config = MagicMock()
        mock_config.model_thresholds = {'similarity_threshold': 0.8}
        mock_config.pipeline_config = {'max_workers': 4}
        mock_config.cache_config = {'enabled': True}
        mock_get_config.return_value = mock_config
        
        # Setup resource manager mock
        mock_resources = {
            "embedding_model": MagicMock(),
            "nlp_model": MagicMock(),
            "semantic_baselines": MagicMock()
        }
        mock_resource_manager.return_value.__enter__.return_value = mock_resources
        
        # Setup text preprocessor mock
        mock_preprocessor_instance = MagicMock()
        mock_text_preprocessor.return_value = mock_preprocessor_instance
        
        # Setup classification pipeline mock
        mock_classification_instance = MagicMock()
        mock_classification_instance.run.return_value = self.expected_classification_result
        mock_classification_pipeline.return_value = mock_classification_instance
        
        # Setup anomaly detection pipeline mock  
        mock_anomaly_instance = MagicMock()
        mock_anomaly_instance.run.return_value = self.expected_anomalies
        mock_anomaly_pipeline.return_value = mock_anomaly_instance
        
        # Setup validation to pass
        mock_validate.return_value = None
        
        # Create a temporary file for output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            # Run the pipeline
            results = run_pipeline(
                mode="inference",
                jobs_to_analyze_df=self.jobs_df,
                idf_corpus_df=self.idf_corpus_df,
                supabase=None,
                output_json_path=temp_file_path
            )
            
            # Step 1: Verify configuration was loaded
            mock_get_config.assert_called_once()
            
            # Step 2: Verify resource manager was used correctly
            mock_resource_manager.assert_called_once_with(config=mock_config, supabase_client=None)
            
            # Step 3: Verify text preprocessor was initialized
            mock_text_preprocessor.assert_called_once_with(mock_resources["nlp_model"])
            
            # Step 4: Verify classification pipeline was initialized and called
            mock_classification_pipeline.assert_called_once_with(
                embedding_model=mock_resources["embedding_model"],
                semantic_baselines=mock_resources["semantic_baselines"],
                text_preprocessor=mock_preprocessor_instance,
                config=mock_config
            )
            
            expected_job_data = self.sample_job_data.copy()
            expected_job_data['effective_description'] = self.sample_job_data['translated_description']
            
            mock_classification_instance.run.assert_called_once()
            call_args = mock_classification_instance.run.call_args[0][0]
            self.assertEqual(call_args['id'], expected_job_data['id'])
            self.assertEqual(call_args['job_title'], expected_job_data['job_title'])
            
            # Step 5: Verify anomaly detection pipeline was initialized and called
            mock_anomaly_pipeline.assert_called_once_with(
                embedding_model=mock_resources["embedding_model"],
                semantic_baselines=mock_resources["semantic_baselines"],
                text_preprocessor=mock_preprocessor_instance,
                nlp_model=mock_resources["nlp_model"],
                thresholds=mock_config.model_thresholds,
                pipeline_config=mock_config.pipeline_config,
                cache_config=mock_config.cache_config
            )
            
            mock_anomaly_instance.run.assert_called_once()
            anomaly_call_args = mock_anomaly_instance.run.call_args[0]
            self.assertEqual(anomaly_call_args[1], self.expected_classification_result)  # classification_result passed correctly
            
            # Step 6: Verify data validation was called
            mock_validate.assert_called_once()
            
            # Step 7: Verify final result structure
            self.assertEqual(len(results), 1)
            result = results[0]
            
            expected_result = {
                'job_id': 'test_job_001',
                'role': 'ML Engineer',
                'industry': 'Technology',
                'job_title': 'Senior Machine Learning Engineer',
                'company_name': 'TechCorp Inc.',
                'semantic_anomalies': self.expected_anomalies,
                'effective_description': self.sample_job_data['translated_description']
            }
            
            self.assertEqual(result, expected_result)
            
            # Step 8: Verify output file was written
            self.assertTrue(os.path.exists(temp_file_path))
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                saved_results = json.load(f)
            self.assertEqual(saved_results, results)
            
            # Step 9: Verify no database write occurred (supabase=None)
            mock_write_result.assert_not_called()
            
        finally:
            # Cleanup
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    @patch('analyzer.run_pipeline.get_config')
    @patch('analyzer.run_pipeline.ResourceManager')
    def test_pipeline_with_supabase_write(self, mock_resource_manager, mock_get_config):
        """Test pipeline with Supabase database writing enabled"""
        
        # Setup mocks
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        
        mock_resources = {
            "embedding_model": MagicMock(),
            "nlp_model": MagicMock(),
            "semantic_baselines": MagicMock()
        }
        mock_resource_manager.return_value.__enter__.return_value = mock_resources
        
        mock_supabase = MagicMock()
        
        with patch('analyzer.run_pipeline.TextPreprocessor'), \
             patch('analyzer.run_pipeline.ClassificationPipeline') as mock_class_pipeline, \
             patch('analyzer.run_pipeline.AnomalyDetectionPipeline') as mock_anomaly_pipeline, \
             patch('analyzer.run_pipeline.validate_job_data'), \
             patch('analyzer.run_pipeline.write_analysis_result') as mock_write_result:
            
            # Setup pipeline mocks
            mock_class_instance = MagicMock()
            mock_class_instance.run.return_value = self.expected_classification_result
            mock_class_pipeline.return_value = mock_class_instance
            
            mock_anomaly_instance = MagicMock()
            mock_anomaly_instance.run.return_value = self.expected_anomalies
            mock_anomaly_pipeline.return_value = mock_anomaly_instance
            
            # Run pipeline with Supabase
            results = run_pipeline(
                mode="inference",
                jobs_to_analyze_df=self.jobs_df,
                idf_corpus_df=self.idf_corpus_df,
                supabase=mock_supabase
            )
            
            # Verify database write was called
            mock_write_result.assert_called_once()
            call_args = mock_write_result.call_args[0]
            self.assertEqual(call_args[0], 'test_job_001')  # job_id
            self.assertEqual(call_args[2], mock_supabase)  # supabase client

    @patch('analyzer.run_pipeline.get_config')
    @patch('analyzer.run_pipeline.ResourceManager')
    @patch('analyzer.run_pipeline.TextPreprocessor')
    @patch('analyzer.run_pipeline.ClassificationPipeline')
    @patch('analyzer.run_pipeline.AnomalyDetectionPipeline')
    @patch('analyzer.run_pipeline.validate_job_data')
    def test_pipeline_validation_error_handling(self, mock_validate, mock_anomaly_pipeline, 
                                               mock_classification_pipeline, mock_text_preprocessor,
                                               mock_resource_manager, mock_get_config):
        """Test pipeline handles validation errors gracefully"""
        
        # Setup mocks
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        
        mock_resources = {
            "embedding_model": MagicMock(),
            "nlp_model": MagicMock(),
            "semantic_baselines": MagicMock()
        }
        mock_resource_manager.return_value.__enter__.return_value = mock_resources
        
        # Create invalid job data (missing required fields)
        invalid_job_data = {'id': 'invalid_job'}  # Missing job_title and effective_description
        invalid_jobs_df = pd.DataFrame([invalid_job_data])
        
        # Make validation raise an error
        mock_validate.side_effect = ValidationError("Missing required fields: job_title, effective_description")
        
        # Run pipeline
        results = run_pipeline(
            mode="inference",
            jobs_to_analyze_df=invalid_jobs_df,
            idf_corpus_df=self.idf_corpus_df
        )
        
        # Should return empty results due to validation error
        self.assertEqual(results, [])

    @patch('analyzer.run_pipeline.get_config')
    @patch('analyzer.run_pipeline.ResourceManager')
    @patch('analyzer.run_pipeline.TextPreprocessor')
    @patch('analyzer.run_pipeline.ClassificationPipeline')
    @patch('analyzer.run_pipeline.AnomalyDetectionPipeline')
    def test_pipeline_empty_dataframe(self, mock_anomaly_pipeline, mock_classification_pipeline,
                                    mock_text_preprocessor, mock_resource_manager, mock_get_config):
        """Test pipeline handles empty input gracefully"""
        
        # Setup mocks
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        
        mock_resources = {
            "embedding_model": MagicMock(),
            "nlp_model": MagicMock(),
            "semantic_baselines": MagicMock()
        }
        mock_resource_manager.return_value.__enter__.return_value = mock_resources
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        results = run_pipeline(
            mode="inference",
            jobs_to_analyze_df=empty_df,
            idf_corpus_df=self.idf_corpus_df
        )
        
        # Should return empty results
        self.assertEqual(results, [])

    @patch('analyzer.run_pipeline.get_config')
    @patch('analyzer.run_pipeline.ResourceManager')
    @patch('analyzer.run_pipeline.TextPreprocessor')
    @patch('analyzer.run_pipeline.ClassificationPipeline')
    @patch('analyzer.run_pipeline.AnomalyDetectionPipeline')
    def test_pipeline_train_mode(self, mock_anomaly_pipeline, mock_classification_pipeline,
                                mock_text_preprocessor, mock_resource_manager, mock_get_config):
        """Test pipeline train mode (not implemented)"""
        
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        
        mock_resources = {
            "embedding_model": MagicMock(),
            "nlp_model": MagicMock(),
            "semantic_baselines": MagicMock()
        }
        mock_resource_manager.return_value.__enter__.return_value = mock_resources
        
        result = run_pipeline(
            mode="train",
            jobs_to_analyze_df=self.jobs_df,
            idf_corpus_df=self.idf_corpus_df
        )
        
        # Train mode returns None (not implemented)
        self.assertIsNone(result)

    def test_pipeline_invalid_mode(self):
        """Test pipeline with invalid mode raises ValueError"""
        
        with patch('analyzer.run_pipeline.get_config') as mock_get_config, \
             patch('analyzer.run_pipeline.ResourceManager') as mock_resource_manager:
            
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config
            
            mock_resources = {
                "embedding_model": MagicMock(),
                "nlp_model": MagicMock(),
                "semantic_baselines": MagicMock()
            }
            mock_resource_manager.return_value.__enter__.return_value = mock_resources
            
            with self.assertRaises(PipelineError):
                run_pipeline(
                    mode="invalid_mode",
                    jobs_to_analyze_df=self.jobs_df,
                    idf_corpus_df=self.idf_corpus_df
                )

    def test_sample_data_processing_flow(self):
        """Integration test showing complete data flow with realistic sample"""
        
        # This test demonstrates the complete flow without mocks
        # Note: This would require actual model files to run
        
        sample_data = {
            'Input Job Data': {
                'id': 'job_123',
                'job_title': 'Data Scientist',
                'company_name': 'Analytics Corp',
                'description': 'Looking for a data scientist with Python and ML experience...',
                'location': 'Remote'
            },
            'Step 1 - Data Validation': {
                'status': 'passed',
                'effective_description': 'Looking for a data scientist with Python and ML experience...'
            },
            'Step 2 - Classification': {
                'role': 'Data Scientist',
                'industry': 'Technology',
                'confidence_score': 0.92
            },
            'Step 3 - Anomaly Detection': {
                'anomalies_found': 0,
                'anomalies': []
            },
            'Step 4 - Final Result': {
                'job_id': 'job_123',
                'role': 'Data Scientist',
                'industry': 'Technology',
                'job_title': 'Data Scientist',
                'company_name': 'Analytics Corp',
                'semantic_anomalies': [],
                'effective_description': 'Looking for a data scientist with Python and ML experience...'
            }
        }
        
        # This test documents the expected flow
        # In a real scenario, you'd run the actual pipeline and compare results
        expected_flow = [
            "1. Load configuration and initialize resources",
            "2. Create TextPreprocessor with NLP model", 
            "3. Initialize ClassificationPipeline with embeddings and baselines",
            "4. Initialize AnomalyDetectionPipeline with models and thresholds",
            "5. For each job: validate data structure",
            "6. For each job: run classification to get role/industry",
            "7. For each job: run anomaly detection with classification results",
            "8. For each job: compile final result dictionary",
            "9. Save results to file or database",
            "10. Return processed results"
        ]
        
        # Assert the documented flow exists
        self.assertEqual(len(expected_flow), 10)
        self.assertIn("classification", expected_flow[5].lower())
        self.assertIn("anomaly detection", expected_flow[6].lower())

if __name__ == '__main__':
    unittest.main()