import logging
import pandas as pd
from analyzer.model import EnhancedJobAnomalyDetector

logger = logging.getLogger(__name__)

def run_pipeline(mode, jobs_df, supabase=None):
    detector = EnhancedJobAnomalyDetector()
    if mode == "train":
        # 预留训练流程
        logger.info("Training mode not implemented yet.")
        return None
    elif mode == "inference":
        detector.calculate_global_idf(jobs_df)
        jobs_df['industry'] = jobs_df.apply(
            lambda row: detector.classify_job_industry(
                row.get('company_name', ''),
                row.get('job_title', ''),
                row.get('description', '')
            ), axis=1
        )
        results = []
        for idx, row in jobs_df.iterrows():
            try:
                job_id = row['id'] if 'id' in row else idx
                company = row.get('company_name', 'Unknown')
                title = row.get('job_title', 'Unknown')
                description = row.get('description', '')
                industry = row.get('industry', 'general')
                specialist_corpus = jobs_df[(jobs_df.index != idx) & (jobs_df['industry'] == industry)]['description'].tolist()
                general_corpus = jobs_df[(jobs_df.index != idx) & (jobs_df['industry'] != industry)]['description'].tolist()
                if len(specialist_corpus) < 2 or len(general_corpus) < 2:
                    logger.warning(f"Skipping {company}: not enough corpus (specialist: {len(specialist_corpus)}, general: {len(general_corpus)})")
                    continue
                anomalies = detector.detect_anomalies_dual_corpus(
                    description, specialist_corpus, general_corpus,
                    job_title=title, company_name=company, industry=industry
                )
                result = {
                    'dual_corpus_anomalies': anomalies,
                    'industry': industry,
                    'company_name': company,
                    'job_title': title,
                    'description': description,
                }
                results.append(result)
                if supabase is not None and 'id' in row:
                    # 可选：写入Supabase
                    from analyzer.inference import write_analysis_result
                    write_analysis_result(row['id'], result, supabase)
            except Exception as e:
                logger.error(f"Error processing job {row.get('id', idx)}: {e}")
        return results
    else:
        raise ValueError(f"Unknown mode: {mode}")
