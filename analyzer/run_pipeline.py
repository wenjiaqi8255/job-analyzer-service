import logging
import pandas as pd
import json
from .utils.exceptions import JobProcessingError, PipelineError
from .utils.data_validator import validate_job_data, ValidationError
from .config import get_config
from .utils.resource_manager import ResourceManager
from .utils.text_preprocessor import TextPreprocessor
from .baseline_classifier.analysis_pipeline import AnalysisPipeline
from .anomaly_detector.anomaly_detection_pipeline import AnomalyDetectionPipeline
from .utils.etl import write_analysis_result
from .content_separator.job_content_separator import JobContentSeparator
from extractor.job_data_accessor import JobDataAccessor

logger = logging.getLogger(__name__)

def run_pipeline(mode, jobs_to_analyze_df, idf_corpus_df, supabase=None, output_json_path='sample_data/output.json'):
    try:
        app_config = get_config()
        with ResourceManager(config=app_config, supabase_client=supabase) as resources:
            
            embedding_model = resources["embedding_model"]
            nlp_model = resources["nlp_model"]
            semantic_baselines = resources["semantic_baselines"]
            boilerplate_baseline = resources["boilerplate_baseline"]
            
            if not embedding_model or not nlp_model:
                logger.error("Failed to load necessary models. Exiting pipeline.")
                return []

            text_preprocessor = TextPreprocessor(nlp_model)
            job_content_separator = JobContentSeparator(nlp_model)
            
            analysis_pipeline = AnalysisPipeline(
                embedding_model=embedding_model,
                semantic_baselines=semantic_baselines,
                text_preprocessor=text_preprocessor,
                config=app_config
            )
            
            anomaly_detection_pipeline = AnomalyDetectionPipeline(
                embedding_model=embedding_model,
                semantic_baselines=semantic_baselines,
                boilerplate_baseline=boilerplate_baseline,
                text_preprocessor=text_preprocessor,
                nlp_model=nlp_model,
                thresholds=app_config.model_thresholds,
                pipeline_config=app_config.pipeline_config,
                cache_config=app_config.cache_config
            )

            if len(jobs_to_analyze_df) == 0:
                logger.warning("No jobs to process")
                return []
            
            logger.info(f"Starting {mode} mode with {len(jobs_to_analyze_df)} jobs to analyze.")
            
            if mode == "train":
                logger.info("Training mode not implemented yet.")
                return None
            elif mode == "inference":
                jobs_to_analyze_df['effective_description'] = jobs_to_analyze_df.apply(JobDataAccessor.get_effective_description, axis=1)

                results = []
                error_count = 0
                
                for idx, row in jobs_to_analyze_df.iterrows():
                    job_id = row.get('id', f'index_{idx}')
                    try:
                        job_data = row.to_dict()
                        validate_job_data(job_data)
                        
                        # Content separation
                        original_description = job_data.get('description', '')
                        if original_description:
                            separated_result = job_content_separator.extract_pure_job_requirements(original_description)
                            job_data['effective_description'] = separated_result['job_content_filtered']
                            job_data['company_description'] = separated_result['removed_company_info']
                        else:
                            job_data['effective_description'] = ''
                            job_data['company_description'] = ''

                        # Run baseline analysis
                        analysis_results = analysis_pipeline.analyze(job_data)
                        job_data.update(analysis_results)

                        # Run anomaly detection
                        anomaly_results = anomaly_detection_pipeline.run_anomaly_detection(
                            job_description=job_data.get('effective_description', ''),
                            analysis_result=job_data
                        )
                        job_data.update(anomaly_results)

                        result = {
                            'job_id': job_id,
                            'role': job_data.get('role', 'general'),
                            'industry': job_data.get('industry', 'unknown'),
                            'job_title': job_data.get('job_title', 'Unknown'),
                            'company_name': job_data.get('company_name', 'Unknown'),
                            'semantic_anomalies': job_data.get('semantic_anomalies', []),
                            'effective_description': job_data.get('effective_description', ''),
                            'company_description': job_data.get('company_description', ''),
                            'role_similarity_analysis': job_data.get('role_similarity_analysis', {}),
                            'industry_similarity_analysis': job_data.get('industry_similarity_analysis', {}),
                            'baseline_composition': job_data.get('baseline_composition', {})
                        }
                        results.append(result)
                        
                        if supabase:
                            write_analysis_result(job_id, result, supabase)

                    except ValidationError as e:
                        error_count += 1
                        logger.warning(f"Skipping job {job_id} due to validation error: {e}")
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
                        if error_count > max(5, len(jobs_to_analyze_df) * 0.3):
                            logger.error("Error rate exceeded threshold, stopping pipeline.")
                            raise PipelineError("High error rate in job processing") from e

                if supabase is None:
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    logger.info(f"Results saved to {output_json_path}")
                    
                return results
            else:
                raise ValueError(f"Unknown mode: {mode}")

    except PipelineError as e:
        logger.error(f"A pipeline failure occurred: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected and unhandled error occurred in the pipeline: {e}", exc_info=True)
        raise PipelineError("An unexpected pipeline failure occurred") from e
