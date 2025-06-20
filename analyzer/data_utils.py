import pandas as pd
import logging
from datetime import datetime, timedelta, timezone
import json
from collections import Counter
from typing import List, Dict
from supabase import Client
import logging
logger = logging.getLogger(__name__)

# We need to import the model to use the preprocessing function for archiving
from analyzer.model import EnhancedJobAnomalyDetector
from utils.job_keyword_extractor import KeywordExtractor

def load_and_process_jobs(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ 成功加载 {len(df)} 条职位数据")
    except Exception as e:
        print(f"❌ 读取CSV文件失败: {e}")
        return pd.DataFrame()
    tech_keywords_high = [
        'software', 'developer', 'programming', 'python', 'java', 'javascript',
        'react', 'vue', 'angular', 'node.js', 'machine learning', 'data science',
        'ai ', 'ml ', 'devops', 'cloud', 'aws', 'azure', 'docker', 'kubernetes'
    ]
    tech_keywords_medium = [
        'technical', 'technology', 'it ', 'computer', 'digital', 'web',
        'mobile', 'app', 'api', 'database', 'sql', 'analytics', 'algorithm'
    ]
    def calculate_tech_score(row):
        title = str(row.get('job_title', '')).lower()
        desc = str(row.get('description', '')).lower()
        combined = title + ' ' + desc
        high_score = sum(2 for keyword in tech_keywords_high if keyword in combined)
        medium_score = sum(1 for keyword in tech_keywords_medium if keyword in combined)
        return high_score + medium_score
    df['tech_score'] = df.apply(calculate_tech_score, axis=1)
    tech_jobs = df[df['tech_score'] >= 2].copy()
    print(f"✅ 筛选出 {len(tech_jobs)} 个技术相关职位")
    return tech_jobs

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

    thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
    
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

def archive_jobs_to_keywords(supabase, days_old=7, batch_size=100):
    """
    Finds jobs older than `days_old`, extracts categorized keywords,
    and copies them to the archive table. It does NOT delete the original records.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting keyword archiving for jobs older than {days_old} days.")

    extractor = KeywordExtractor()

    # 1. Fetch jobs from `job_listings` that are older than `days_old`
    archive_date = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
    response = supabase.table("job_listings").select("id").lte("created_at", archive_date).execute()
    old_job_ids = {job['id'] for job in response.data}

    if not old_job_ids:
        logger.info("No jobs older than 7 days found to archive.")
        return 0

    # 2. Find out which of them are already in the archive to avoid duplicates
    response = supabase.table("job_analytics_archive").select("job_id").in_("job_id", list(old_job_ids)).execute()
    archived_job_ids = {job['job_id'] for job in response.data}
    
    jobs_to_archive_ids = list(old_job_ids - archived_job_ids)
    
    if not jobs_to_archive_ids:
        logger.info("All old jobs have already been archived.")
        return 0

    logger.info(f"Found {len(jobs_to_archive_ids)} new jobs to archive.")

    # 3. Process in batches
    archived_count = 0
    for i in range(0, len(jobs_to_archive_ids), batch_size):
        batch_ids = jobs_to_archive_ids[i:i+batch_size]
        response = supabase.table("job_listings").select("*").in_("id", batch_ids).execute()
        jobs_batch = response.data

        archive_records = []
        for job in jobs_batch:
            description = job.get('description', '')
            if not description:
                continue
            
            extracted_data = extractor.extract_keywords(description)
            archive_record = {
                "job_id": job['id'],
                "company_name": job.get('company_name', '')[:50],
                "job_title": job.get('job_title', '')[:50],
                "industry": job.get('industry', '')[:50],
                "created_at": job.get('created_at'),
                "archived_from_date": job.get('created_at'),
                "technical_keywords": json.dumps(extracted_data.get('technical', {})),
                "domain_keywords": json.dumps(extracted_data.get('domain', {})),
                "tools_keywords": json.dumps(extracted_data.get('business_tools', {})),
                "soft_skills_keywords": json.dumps(extracted_data.get('soft_skills', {})),
                "requirements_keywords": json.dumps(extracted_data.get('job_requirements', {})),
            }
            archive_records.append(archive_record)
        
        if archive_records:
            try:
                insert_response = supabase.table("job_analytics_archive").insert(archive_records).execute()
                if len(insert_response.data) > 0:
                     archived_count += len(insert_response.data)
                     logger.info(f"Successfully archived {len(insert_response.data)} jobs' keywords.")
                else:
                    logger.error(f"Failed to archive jobs. Supabase response: {insert_response.error or 'No data returned'}")

            except Exception as e:
                logger.error(f"Error inserting into Supabase archive: {e}")
                logger.info("Saving failing batch to failed_archive_data.json for debugging.")
                try:
                    with open('failed_archive_data.json', 'w') as f:
                        json.dump(archive_records, f, indent=2, default=str)
                except Exception as file_e:
                    logger.error(f"Could not write failure log to file: {file_e}")

    logger.info(f"Keyword archiving complete. Total jobs processed: {archived_count}.")
    return archived_count

def archive_all_jobs_to_keywords(supabase, batch_size=100):
    """
    Finds all jobs in the job_listings table that have not yet been archived,
    extracts their keywords, and saves them to the archive table.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting full keyword archiving for ALL jobs.")

    extractor = KeywordExtractor()

    # 1. Fetch all job IDs from `job_listings`
    response = supabase.table("job_listings").select("id", count='exact').execute()
    all_job_ids = {job['id'] for job in response.data}
    logger.info(f"Found a total of {response.count} jobs in job_listings.")

    if not all_job_ids:
        logger.info("No jobs found in job_listings to archive.")
        return 0

    # 2. Find out which of them are already in the archive to avoid duplicates
    # We need to do this in batches if all_job_ids is very large, as `in_` has limits.
    archived_job_ids = set()
    all_job_ids_list = list(all_job_ids)
    for i in range(0, len(all_job_ids_list), 500): # Check 500 at a time
        batch_ids = all_job_ids_list[i:i+500]
        resp = supabase.table("job_analytics_archive").select("job_id").in_("job_id", batch_ids).execute()
        archived_job_ids.update({job['job_id'] for job in resp.data})

    jobs_to_archive_ids = list(all_job_ids - archived_job_ids)
    
    if not jobs_to_archive_ids:
        logger.info("All jobs have already been archived.")
        return 0

    logger.info(f"Found {len(jobs_to_archive_ids)} new jobs to archive.")

    # 3. Process in batches
    archived_count = 0
    for i in range(0, len(jobs_to_archive_ids), batch_size):
        batch_ids = jobs_to_archive_ids[i:i+batch_size]
        response = supabase.table("job_listings").select("*").in_("id", batch_ids).execute()
        jobs_batch = response.data

        archive_records = []
        for job in jobs_batch:
            description = job.get('description', '')
            if not description:
                continue
            
            extracted_data = extractor.extract_keywords(description)
            archive_record = {
                "job_id": job['id'],
                "company_name": job.get('company_name', '')[:50],
                "job_title": job.get('job_title', '')[:50],
                "industry": job.get('industry', '')[:50],
                "created_at": job.get('created_at'),
                "archived_from_date": job.get('created_at'),
                "technical_keywords": json.dumps(extracted_data.get('technical', {})),
                "domain_keywords": json.dumps(extracted_data.get('domain', {})),
                "tools_keywords": json.dumps(extracted_data.get('business_tools', {})),
                "soft_skills_keywords": json.dumps(extracted_data.get('soft_skills', {})),
                "requirements_keywords": json.dumps(extracted_data.get('job_requirements', {})),
            }
            archive_records.append(archive_record)
        
        if archive_records:
            try:
                insert_response = supabase.table("job_analytics_archive").insert(archive_records).execute()
                if len(insert_response.data) > 0:
                     archived_count += len(insert_response.data)
                     logger.info(f"Successfully archived {len(insert_response.data)} jobs' keywords.")
                else:
                    logger.error(f"Failed to archive jobs. Supabase response: {insert_response.error or 'No data returned'}")

            except Exception as e:
                logger.error(f"Error inserting into Supabase archive: {e}")
                logger.info("Saving failing batch to failed_archive_data.json for debugging.")
                try:
                    with open('failed_archive_data.json', 'w') as f:
                        json.dump(archive_records, f, indent=2, default=str)
                except Exception as file_e:
                    logger.error(f"Could not write failure log to file: {file_e}")

    logger.info(f"Full keyword archiving complete. Total jobs processed: {archived_count}.")
    return archived_count

def cleanup_job_listings(supabase, days_old=8, batch_size=100):
    """
    Deletes jobs from `job_listings` that are older than `days_old`,
    but only if they have been safely archived first.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting cleanup of job_listings for jobs older than {days_old} days.")

    # 1. Find jobs older than `days_old` in the main table
    delete_date = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
    response = supabase.table("job_listings").select("id").lte("created_at", delete_date).execute()
    job_ids_to_delete = [job['id'] for job in response.data]

    if not job_ids_to_delete:
        logger.info("No jobs older than 8 days to clean up.")
        return 0

    logger.info(f"Found {len(job_ids_to_delete)} potential jobs to delete. Verifying they are archived...")
    
    deleted_count = 0
    for i in range(0, len(job_ids_to_delete), batch_size):
        batch_ids = job_ids_to_delete[i:i+batch_size]

        # 2. Confirm these jobs exist in the archive (Safety Check)
        try:
            archive_response = supabase.table("job_analytics_archive").select("job_id").in_("job_id", batch_ids).execute()
            confirmed_ids = {job['job_id'] for job in archive_response.data}
            
            if not confirmed_ids:
                logger.warning(f"Found {len(batch_ids)} old jobs, but none were confirmed in the archive. Skipping deletion for this batch.")
                continue

            # 3. Perform the deletion
            delete_response = supabase.table("job_listings").delete().in_("id", list(confirmed_ids)).execute()

            if len(delete_response.data) > 0:
                count = len(delete_response.data)
                deleted_count += count
                logger.info(f"Successfully deleted {count} jobs from job_listings.")
            else:
                logger.warning(f"Deletion command for {len(confirmed_ids)} jobs completed, but no rows were returned. Maybe they were already deleted. Error: {delete_response.error}")

        except Exception as e:
            logger.error(f"An error occurred during the cleanup of batch starting at index {i}: {e}", exc_info=True)

    logger.info(f"job_listings cleanup complete. Total jobs deleted: {deleted_count}.")
    return deleted_count

def cleanup_archive_table(supabase, days_to_keep=180):
    """
    Removes records from job_analytics_archive older than `days_to_keep`.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Cleaning up archive table, removing records older than {days_to_keep} days.")
    
    cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
    
    try:
        response = supabase.table("job_analytics_archive").delete().lte("created_at", cutoff_date).execute()
        
        if response.data:
            deleted_count = len(response.data)
            logger.info(f"Successfully deleted {deleted_count} old records from archive.")
            return deleted_count
        else:
            # Note: A successful delete with 0 rows affected also returns an empty list in `data`.
            logger.info("No old records found to delete, or deletion failed silently.")
            return 0
            
    except Exception as e:
        logger.error(f"Error during archive cleanup: {e}", exc_info=True)
        return -1 # Indicate error