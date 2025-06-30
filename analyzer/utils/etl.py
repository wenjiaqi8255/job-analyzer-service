import logging
from datetime import datetime, timezone
from supabase import Client
import json

logger = logging.getLogger(__name__)

def write_analysis_result(job_id: str, analysis_result: dict, supabase: Client):
    """Upserts the analysis result into the job_anomaly_analysis table."""
    
    # For backward compatibility and data integrity, ensure scores are float, not float32
    role_similarities = {k: float(v) for k, v in analysis_result.get('role_similarity_analysis', {}).items()}
    industry_similarities = {k: float(v) for k, v in analysis_result.get('industry_similarity_analysis', {}).items()}

    # Construct the payload according to the new schema design
    payload = {
        "job_listing_id": job_id,
        "last_analyzed_at": datetime.now(timezone.utc).isoformat(),
        "role_similarities": json.dumps(role_similarities),
        "industry_similarities": json.dumps(industry_similarities),
        "analysis_summary": json.dumps({
            "primary_role": analysis_result.get('role'),
            "primary_industry": analysis_result.get('industry'),
            "confidence_level": "high",  # Placeholder, logic to be implemented
            "cross_domain_indicator": False  # Placeholder, logic to be implemented
        }),
        "analysis_metadata": json.dumps({
            "computation_time_ms": -1, # Placeholder
        }),
        "baseline_composition": json.dumps(analysis_result.get('baseline_composition', {})), # Placeholder for future enhancement
        "analysis_data": json.dumps(analysis_result) # Store the full analysis for debugging/future use
    }

    try:
        # Use upsert to either insert a new record or update an existing one
        response = supabase.table("job_anomaly_analysis").upsert(payload).execute()
        
        # Also mark the job as processed
        supabase.table("job_listings").update({
            "processed_for_matching": True,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", job_id).execute()
        
        if response.data:
            logger.info(f"Successfully wrote analysis for job {job_id}")
        elif response.error:
            logger.error(f"Failed to write analysis for job {job_id}: {response.error.message}")
        else:
            logger.warning(f"Analysis for job {job_id} write operation returned no data and no error.")

    except Exception as e:
        logger.error(f"Error writing analysis for job {job_id}: {e}", exc_info=True)