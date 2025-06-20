import logging
import pandas as pd
import json
import hashlib
from datetime import datetime, timezone
from analyzer.model import EnhancedJobAnomalyDetector

logger = logging.getLogger(__name__)

def _get_corpus_version_hash(df):
    """Creates a unique hash based on the IDs and count of jobs in the corpus."""
    if df.empty:
        return "empty_corpus"
    # Use job_id for archive data, id for listings data
    id_col = 'job_id' if 'job_id' in df.columns else 'id'
    id_string = "".join(sorted(df[id_col].astype(str).tolist()))
    hash_input = f"{len(df)}-{id_string}"
    return hashlib.md5(hash_input.encode('utf-8')).hexdigest()

def run_pipeline(mode, jobs_to_analyze_df, idf_corpus_df, supabase=None, output_json_path='sample_data/output.json'):
    detector = EnhancedJobAnomalyDetector()
    
    if len(jobs_to_analyze_df) == 0:
        logger.warning("No jobs to process")
        return []
    
    logger.info(f"Starting {mode} mode with {len(jobs_to_analyze_df)} jobs to analyze, using a corpus of {len(idf_corpus_df)} jobs for IDF.")
    
    detector = EnhancedJobAnomalyDetector()
    if mode == "train":
        # 预留训练流程
        logger.info("Training mode not implemented yet.")
        return None
    elif mode == "inference":
        # === IDF Caching Logic ===
        version_hash = _get_corpus_version_hash(idf_corpus_df)
        logger.info(f"Corpus version hash: {version_hash}")
        cache_loaded = False
        
        # 1. Try to load from cache
        if supabase:
            try:
                response = supabase.table("idf_cache").select("*").eq("data_version_hash", version_hash).execute()
                if response.data:
                    logger.info("✅ Found valid IDF cache in Supabase. Loading...")
                    cached_idf = {item['word']: item['idf_value'] for item in response.data}
                    detector.set_idf_cache(cached_idf)
                    cache_loaded = True
                else:
                    logger.info("No valid cache found. Calculating new IDF...")
                    detector.calculate_global_idf(idf_corpus_df)
                    cache_loaded = False  # 明确设置
            except Exception as e:
                logger.warning(f"Could not reach IDF cache. Calculating new IDF. Error: {e}")
                detector.calculate_global_idf(idf_corpus_df)
                cache_loaded = False  # 明确设置
        else:
             # Fallback for local mode
            detector.calculate_global_idf(idf_corpus_df)

        # 2. If it was just calculated, save to cache
        idf_cache_data = detector.get_idf_cache()
        if supabase and idf_cache_data and not cache_loaded:
            logger.info(f"Saving {len(idf_cache_data)} new IDF values to cache...")
            records_to_insert = [
                {
                    "word": word,
                    "idf_value": float(idf_value),
                    "data_version_hash": version_hash,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "category": "global" # Placeholder for now
                }
                for word, idf_value in idf_cache_data.items()
            ]
            try:
                # Simple approach: delete old and insert new. A proper upsert is better.
                # For now, we assume hashes are unique enough that we don't have to delete.
                supabase.table("idf_cache").insert(records_to_insert).execute()
            except Exception as e:
                logger.error(f"Failed to save IDF cache to Supabase: {e}")

        # Classify industry for BOTH dataframes to ensure consistency
        idf_corpus_df['industry'] = idf_corpus_df.apply(
            lambda row: detector.classify_job_industry(
                row.get('company_name', ''),
                row.get('job_title', ''),
                row.get('description', '')
            ), axis=1
        )
        jobs_to_analyze_df['industry'] = jobs_to_analyze_df.apply(
            lambda row: detector.classify_job_industry(
                row.get('company_name', ''),
                row.get('job_title', ''),
                row.get('description', '')
            ), axis=1
        )

        results = []
        success_count = 0
        error_count = 0
        for idx, row in jobs_to_analyze_df.iterrows():
            try:
                job_id = row['id']
                company = row.get('company_name', 'Unknown')
                title = row.get('job_title', 'Unknown')
                description = row.get('description', '')
                industry = row.get('industry', 'general')

                # Build specialist and general corpuses from the historical IDF data
                specialist_corpus = idf_corpus_df[idf_corpus_df['industry'] == industry]['description'].tolist()
                general_corpus = idf_corpus_df[idf_corpus_df['industry'] != industry]['description'].tolist()
                success_count += 1

                if len(specialist_corpus) < 2 or len(general_corpus) < 2:
                    # It's better to classify the corpus first to avoid this warning for every job.
                    # For now, we keep it simple.
                    logger.warning(f"Skipping {company} - {title}: not enough corpus for industry '{industry}' (specialist: {len(specialist_corpus)}, general: {len(general_corpus)})")
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
                    'description': description, # Keep original description in result
                }
                results.append(result)
                
                if supabase is not None:
                    from analyzer.inference import write_analysis_result
                    write_analysis_result(job_id, result, supabase)
                    
                    # Also mark the job as processed
                    supabase.table("job_listings").update({"processed_for_matching": True}).eq("id", job_id).execute()

            except Exception as e:
                error_count += 1
                logger.error(f"Error processing job {row.get('id', idx)}: {e}")
                # 达到错误阈值时停止
                if error_count > len(jobs_to_analyze_df) * 0.3:  # 30%错误率阈值
                    logger.error("Too many errors, stopping pipeline")
                    break
        
        # If using Supabase, update the records
        if supabase:
            update_payload = [
                {
                    "id": row['id'],
                    "processed_for_matching": True,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "tags": row['tags']
                }
                for _, row in jobs_to_analyze_df.iterrows()
            ]
            try:
                supabase.table("job_listings").upsert(update_payload).execute()
                logger.info(f"Successfully updated {len(update_payload)} records in Supabase.")
            except Exception as e:
                logger.error(f"Failed to update records in Supabase: {e}", exc_info=True)

        # This part is for local CSV testing and can be removed or kept. Let's keep it.
        if supabase is None:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_json_path}")
            
        return results
    else:
        raise ValueError(f"Unknown mode: {mode}")
