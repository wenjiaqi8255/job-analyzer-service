import pandas as pd
import logging
from datetime import datetime, timedelta, timezone
import json
from typing import List, Dict
from supabase import Client

logger = logging.getLogger(__name__)

def _reconstruct_description_from_keywords(row):
    """Helper to reconstruct a text document from keyword JSON columns."""
    text_parts = []
    keyword_columns = [
        'technical_keywords', 'domain_keywords', 'tools_keywords',
        'soft_skills_keywords', 'requirements_keywords'
    ]
    for col in keyword_columns:
        if col in row and row[col]:
            try:
                # Handle both dict and string representations of JSON
                keywords = json.loads(row[col]) if isinstance(row[col], str) else row[col]
                if isinstance(keywords, dict):
                    for word, count in keywords.items():
                        text_parts.extend([word] * count)
            except (json.JSONDecodeError, TypeError):
                continue
    return " ".join(text_parts)

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