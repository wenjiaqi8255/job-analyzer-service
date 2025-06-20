import argparse
import logging
import os
import sys
from dotenv import load_dotenv
import pandas as pd

# Add the project root to the Python path before local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from supabase import create_client, Client
from analyzer.db_queries import fetch_jobs_to_analyze, fetch_idf_corpus
from analyzer.etl import (
    archive_jobs_to_keywords, cleanup_job_listings, cleanup_archive_table,
    archive_all_jobs_to_keywords
)
from analyzer.pipeline import run_pipeline
from playground.local_data_util import load_and_process_jobs

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Job Analyzer Pipeline")
    parser.add_argument('--mode', choices=['train', 'inference'], default='inference', help='Pipeline mode')
    parser.add_argument('--csv', type=str, help='Path to CSV file for inference')
    parser.add_argument('--supabase', action='store_true', help='Use Supabase as data source and sink')
    parser.add_argument('--run-keyword-archiving', action='store_true', help='Run the 7-day keyword archiving process and exit')
    parser.add_argument('--run-listings-cleanup', action='store_true', help='Run the 8-day cleanup of the job_listings table')
    parser.add_argument('--run-archive-cleanup', action='store_true', help='Run the 180-day cleanup of the job_analytics_archive table')
    parser.add_argument('--run-full-keyword-archiving', action='store_true', help='Manually run keyword archiving for all non-archived jobs')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for Supabase inference')
    args = parser.parse_args()

    supabase = None
    
    # Connect to Supabase if any Supabase-related action is requested
    if args.supabase or args.run_keyword_archiving or args.run_listings_cleanup or args.run_archive_cleanup or args.run_full_keyword_archiving:
        if os.path.exists('.env'):
            load_dotenv()
            
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.error("SUPABASE_URL or SUPABASE_KEY not set in environment.")
            sys.exit(1)
            
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("✅ Successfully connected to Supabase")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            sys.exit(1)

    # --- Data Lifecycle Management ---
    if args.run_keyword_archiving:
        logger.info("🚀 Starting 7-day keyword archiving process...")
        archived_count = archive_jobs_to_keywords(supabase)
        logger.info(f"✅ Keyword archiving process completed. Processed {archived_count} jobs.")
        return

    if args.run_full_keyword_archiving:
        logger.info("🚀 Starting full keyword archiving process for all jobs...")
        archived_count = archive_all_jobs_to_keywords(supabase)
        logger.info(f"✅ Full keyword archiving process completed. Processed {archived_count} jobs.")
        return

    if args.run_listings_cleanup:
        logger.info("🚀 Starting 8-day job_listings cleanup process...")
        deleted_count = cleanup_job_listings(supabase)
        logger.info(f"✅ job_listings cleanup process completed. Deleted {deleted_count} jobs.")
        return
        
    if args.run_archive_cleanup:
        logger.info("🚀 Starting 180-day archive cleanup process...")
        deleted_count = cleanup_archive_table(supabase, days_to_keep=180)
        logger.info(f"✅ Archive cleanup process completed. Deleted {deleted_count} records.")
        return

    # --- Main Analysis Pipeline ---
    if not args.supabase and not args.csv:
        logger.error("❌ Please provide an execution mode: --supabase for analysis, or one of the lifecycle tasks like --run-keyword-archiving.")
        sys.exit(1)

    jobs_to_analyze_df = None
    idf_corpus_df = None

    if args.supabase:
        if args.mode == 'inference':
            try:
                idf_corpus_df = fetch_idf_corpus(supabase)
                if idf_corpus_df.empty:
                    logger.error("❌ Could not fetch corpus for IDF. Aborting.")
                    sys.exit(1)

                jobs_to_analyze = fetch_jobs_to_analyze(supabase, batch_size=args.batch_size)
                if not jobs_to_analyze:
                    logger.info("✅ No new jobs to process. All caught up!")
                    return
                    
                jobs_to_analyze_df = pd.DataFrame(jobs_to_analyze)
                logger.info(f"📊 Loaded {len(jobs_to_analyze_df)} jobs for analysis")
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch jobs from Supabase: {e}", exc_info=True)
                sys.exit(1)
                
    elif args.csv:
        # Local CSV mode now needs to handle both dataframes.
        # For simplicity, we'll use the same CSV for both.
        if not os.path.exists(args.csv):
            logger.error(f"❌ CSV file not found: {args.csv}")
            sys.exit(1)
            
        try:
            jobs_df = load_and_process_jobs(args.csv)
            if jobs_df.empty:
                logger.error("❌ No valid jobs loaded from CSV")
                sys.exit(1)
            jobs_to_analyze_df = jobs_df
            idf_corpus_df = jobs_df.copy() # Use the same data for corpus in local mode
        except Exception as e:
            logger.error(f"❌ Failed to load CSV: {e}")
            sys.exit(1)
    else:
        logger.error("❌ Please provide either --csv or --supabase for the main pipeline.")
        sys.exit(1)

    # 运行pipeline
    try:
        logger.info(f"🚀 Starting {args.mode} pipeline...")
        results = run_pipeline(
            args.mode, 
            jobs_to_analyze_df=jobs_to_analyze_df, 
            idf_corpus_df=idf_corpus_df, 
            supabase=supabase
        )
        
        if results:
            logger.info(f"✅ Pipeline completed successfully! Processed {len(results)} jobs.")
        else:
            logger.warning("⚠️ Pipeline completed but no results were generated.")
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()