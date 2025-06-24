import logging
import pandas as pd
import json
from .exceptions import JobProcessingError, PipelineError
from .data_validator import validate_job_data, ValidationError
from .config import get_config
from .resource_manager import ResourceManager
from .text_preprocessor import TextPreprocessor
from .classifier.classification_pipeline import ClassificationPipeline
from .anomaly_detection_pipeline import AnomalyDetectionPipeline
from .inference import write_analysis_result
from extractor.job_data_accessor import JobDataAccessor

logger = logging.getLogger(__name__)

def run_pipeline(mode, jobs_to_analyze_df, idf_corpus_df, supabase=None, output_json_path='sample_data/output.json'):
    try:
        app_config = get_config()
        with ResourceManager(config=app_config, supabase_client=supabase) as resources:
            
            embedding_model = resources["embedding_model"]
            nlp_model = resources["nlp_model"]
            semantic_baselines = resources["semantic_baselines"]
            
            if not embedding_model or not nlp_model:
                logger.error("Failed to load necessary models. Exiting pipeline.")
                return []

            text_preprocessor = TextPreprocessor(nlp_model)
            
            classification_pipeline = ClassificationPipeline(
                embedding_model=embedding_model,
                semantic_baselines=semantic_baselines,
                text_preprocessor=text_preprocessor,
                config=app_config
            )
            
            anomaly_detection_pipeline = AnomalyDetectionPipeline(
                embedding_model=embedding_model,
                semantic_baselines=semantic_baselines,
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

                        classification_result = classification_pipeline.run(job_data)
                        
                        anomalies = anomaly_detection_pipeline.run(job_data, classification_result)
                        
                        result = {
                            'job_id': job_id,
                            'role': classification_result['role'],
                            'industry': classification_result['industry'],
                            'job_title': job_data.get('job_title', 'Unknown'),
                            'company_name': job_data.get('company_name', 'Unknown'),
                            'semantic_anomalies': anomalies,
                            'effective_description': job_data.get('effective_description', '')
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
