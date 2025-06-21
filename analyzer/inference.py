import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def write_analysis_result(job_id, analysis_data, supabase):
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "job_listing_id": job_id,
        "analysis_data": analysis_data,
        "last_analyzed_at": now
    }
    
    response = supabase.table("job_anomaly_analysis").unsert(payload).execute()
    
    if len(response.data) > 0:
        logger.info(f"Successfully wrote analysis for job {job_id}")
    else:
        logger.error(f"Failed to write analysis for job {job_id}: {response.error}")
