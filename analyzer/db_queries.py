import pandas as pd
import logging
from supabase import Client

logger = logging.getLogger(__name__)

def fetch_all_job_listings(supabase: Client) -> pd.DataFrame:
    """
    Fetches all job listings from the Supabase 'job_listings' table.
    
    Args:
        supabase: The Supabase client instance.
        
    Returns:
        A pandas DataFrame containing all job listings, sorted by creation date.
    """
    logger.info("Fetching all jobs from the 'job_listings' table...")
    try:
        response = (
            supabase.table("job_listings")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        
        if response.data:
            logger.info(f"Successfully fetched {len(response.data)} job listings.")
            return pd.DataFrame(response.data)
        else:
            logger.warning("No job listings found in the table.")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"An error occurred while fetching job listings: {e}", exc_info=True)
        # Return an empty DataFrame on failure
        return pd.DataFrame()

def fetch_job_by_id(supabase: Client, job_id: str) -> pd.DataFrame:
    """
    Fetches a single job listing by its ID.
    
    Args:
        supabase: The Supabase client instance.
        job_id: The UUID of the job to fetch.
        
    Returns:
        A pandas DataFrame containing the job listing, or an empty DataFrame if not found.
    """
    logger.info(f"Fetching job by ID: {job_id}...")
    try:
        response = (
            supabase.table("job_listings")
            .select("*")
            .eq("id", job_id)
            .execute()
        )
        if response.data:
            logger.info(f"Successfully fetched job {job_id}.")
            return pd.DataFrame(response.data)
        else:
            logger.warning(f"Job with ID {job_id} not found.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching job by ID {job_id}: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_active_semantic_baselines(supabase_client: Client):
    """
    Fetches all active semantic baselines from the database.
    Active baselines are used to build the in-memory cache for the anomaly detector.
    """
    try:
        logger.info("Fetching active semantic baselines from database...")
        response = supabase_client.table('semantic_baselines').select('*').eq('is_active', True).execute()
        
        if response.data:
            logger.info(f"Successfully fetched {len(response.data)} active baselines.")
            return response.data
        else:
            logger.warning("No active semantic baselines found in the database.")
            return []
    except Exception as e:
        logger.error(f"Error fetching semantic baselines: {e}", exc_info=True)
        return []

# 在 db_queries.py 中
def fetch_baseline_vectors(supabase_client: Client, baseline_id: int, model_name: str):
    """
    Fetches the pre-computed vector record(s) for a specific baseline and embedding model.
    """
    try:
        response = supabase_client.table('baseline_vectors') \
            .select('vectors_data') \
            .eq('baseline_id', baseline_id) \
            .eq('embedding_model', model_name) \
            .limit(1) \
            .execute()
        
        if response.data:
            return response.data  # Return the full list of records, e.g., [{'vectors_data': {...}}]
        else:
            logger.warning(f"No vectors found for baseline_id={baseline_id} and model='{model_name}'")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching baseline vectors for baseline_id={baseline_id}: {e}", exc_info=True)
        return None