import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def write_analysis_result(job_id, analysis_data, supabase):
    now = datetime.utcnow().isoformat()
    payload = {
        "job_listing_id": job_id,
        "analysis_data": analysis_data,
        "last_analyzed_at": now,
    }
    response = supabase.table("job_anomaly_analysis").insert(payload).execute()
    if response.error is None:
        logger.info(f"Analysis written for job {job_id}.")
    else:
        logger.error(f"Failed to write analysis for job {job_id}: {response.error}")
