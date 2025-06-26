import logging
from datetime import datetime, timezone
from supabase import Client

logger = logging.getLogger(__name__)

def write_analysis_result(job_id: str, analysis_data: dict, supabase: Client):
    """Upserts the analysis result into the job_anomaly_analysis table."""
    payload = {
        "job_listing_id": job_id,
        "analysis_data": analysis_data,
        "last_analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        # Use upsert to either insert a new record or update an existing one
        response = supabase.table("job_anomaly_analysis").upsert(payload).execute()
        
        # Also mark the job as processed
        supabase.table("job_listings").update({"processed_for_matching": True}).eq("id", job_id).execute()
        
        if len(response.data) > 0:
            logger.info(f"Successfully wrote analysis for job {job_id}")
        else:
            logger.error(f"Failed to write analysis for job {job_id}: {response.error}")
    except Exception as e:
        logger.error(f"Error writing analysis for job {job_id}: {e}")
