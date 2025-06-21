import pandas as pd
import logging
from datetime import datetime, timedelta, timezone
import json
from typing import List, Dict
from supabase import Client

logger = logging.getLogger(__name__)

def _reconstruct_description_from_keywords(row):
    """
    Helper to reconstruct a text document from the 'keywords' JSONB column,
    which is expected to be a list of [keyword, score] pairs.
    """
    keywords_data = row.get('keywords')
    if not keywords_data:
        return ""
    
    # Handle both JSON string and Python list/dict cases
    if isinstance(keywords_data, str):
        try:
            keywords_data = json.loads(keywords_data)
        except json.JSONDecodeError:
            # If it's not a valid JSON string, cannot process.
            return ""

    if isinstance(keywords_data, list) and keywords_data:
        # Expected format: [['keyword1', 0.8], ['keyword2', 0.7]]
        # We only need the keyword (the first element) for IDF calculation.
        keywords = [item[0] for item in keywords_data if isinstance(item, list) and len(item) > 0]
        return " ".join(keywords)

    return ""

def fetch_idf_corpus(supabase) -> pd.DataFrame:
    """
    Fetches a corpus of jobs for IDF calculation.
    It prioritizes the job_analytics_archive table (last 30 days).
    If the archive is too small, it falls back to the job_listings table.
    """
    logger = logging.getLogger(__name__)
    logger.info("Fetching corpus for IDF calculation...")

    thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    
    # 1. Try to fetch from the archive first
    response = supabase.table("job_analytics_archive").select("*").gte("created_at", thirty_days_ago).execute()
    
    archived_jobs = response.data
    logger.info(f"Found {len(archived_jobs)} jobs in archive from last 30 days.")

    if len(archived_jobs) >= 100:
        logger.info("Using archived jobs for IDF corpus.")
        df = pd.DataFrame(archived_jobs)
        # Reconstruct description for TF-IDF vectorizer
        df['description'] = df.apply(_reconstruct_description_from_keywords, axis=1)
        # Ensure essential columns exist
        df['job_title'] = df.get('job_title', '')
        df['company_name'] = df.get('company_name', '')
        return df
    
    # 2. Fallback to recent job_listings if archive is insufficient
    logger.warning("Archive has less than 100 jobs. Falling back to recent job_listings for IDF corpus.")
    response = supabase.table("job_listings").select("*").order("created_at", desc=True).limit(50).execute()
    
    if not response.data:
        logger.error("No jobs found in job_listings as fallback. Returning empty DataFrame.")
        return pd.DataFrame()
        
    logger.info(f"Using last {len(response.data)} jobs from job_listings for IDF corpus.")
    return pd.DataFrame(response.data)

def fetch_all_idf_corpus(supabase) -> pd.DataFrame:
    """
    Fetches the entire job_analytics_archive for a full IDF calculation.
    """
    logger.info("Fetching full archive for IDF calculation...")
    
    response = supabase.table("job_analytics_archive").select("*").execute()
    
    archived_jobs = response.data
    logger.info(f"Found {len(archived_jobs)} total jobs in archive.")

    if not archived_jobs:
        logger.warning("No jobs found in archive. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(archived_jobs)
    # Reconstruct description for TF-IDF vectorizer
    df['description'] = df.apply(_reconstruct_description_from_keywords, axis=1)
    # Ensure essential columns exist
    df['job_title'] = df.get('job_title', '')
    df['company_name'] = df.get('company_name', '')
    # The 'id' column is needed for processing, but archive table has 'job_id'
    if 'job_id' in df.columns:
        df['id'] = df['job_id']
    return df

def fetch_jobs_to_analyze(supabase: Client, batch_size: int = 50) -> List[Dict]:
    """Fetches jobs that haven't been processed for matching yet."""
    try:
        response = (
            supabase.table("job_listings")
            .select("*")
            .is_("processed_for_matching", "null")
            .order("created_at", desc=True)
            .limit(batch_size)
            .execute()
        )
        
        jobs_to_analyze = response.data
        
        if not jobs_to_analyze:
            logger.info("No new jobs to analyze.")
            return []
        
        logger.info(f"Found {len(jobs_to_analyze)} new jobs to analyze.")
        return jobs_to_analyze
    except Exception as e:
        logger.error(f"Error fetching jobs to analyze: {e}")
        return [] 