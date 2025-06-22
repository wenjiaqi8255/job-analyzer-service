import argparse
import logging
import os
import sys
from dotenv import load_dotenv
import pandas as pd

# Add the project root to the Python path before local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from supabase import create_client, Client
from analyzer.db_queries import fetch_all_job_listings, fetch_job_by_id
from analyzer.pipeline import run_pipeline
from playground.local_data_util import load_and_process_jobs

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Job Analyzer Pipeline")
    parser.add_argument('--mode', choices=['train', 'inference'], default='inference', help='Pipeline mode. Currently only inference is supported.')
    parser.add_argument('--csv', type=str, help='Path to local CSV file for inference.')
    parser.add_argument('--supabase', action='store_true', help='Use Supabase as the data source and sink.')
    parser.add_argument('--analyze-all', action='store_true', help='Analyze all unprocessed jobs in the database.')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for Supabase inference when not using --analyze-all.')
    parser.add_argument('--translate', action='store_true', help='Run translation batch job for untranslated jobs.')
    parser.add_argument('--translate-all', action='store_true', help='Run translation for all German jobs, overwriting existing translations.')
    parser.add_argument('--job-id', type=str, help='Analyze a single job by its ID (UUID). Requires --supabase.')
    args = parser.parse_args()

    supabase = None
    
    if args.supabase or args.translate or args.translate_all:
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

    # --- Translation Batch Process ---
    if args.translate or args.translate_all:
        logger.info("🚀 Starting translation process...")
        from translation.translation_service import TranslationService
        service = TranslationService(supabase_client=supabase)
        if args.translate_all:
            success = service.run_full_translation()
        else:
            success = service.run_translation_batch()
        
        if success:
            logger.info("✅ Translation completed successfully.")
        else:
            logger.error("❌ Translation failed.")
            sys.exit(1)
        return
    
    # --- Main Analysis Pipeline ---
    if not args.supabase and not args.csv and not args.job_id:
        parser.print_help()
        logger.error("\nPlease provide a data source: --supabase, --csv, or --job-id")
        sys.exit(1)

    if args.job_id and not args.supabase:
        logger.error("--job-id requires --supabase to be set.")
        sys.exit(1)

    jobs_to_analyze_df = None
    idf_corpus_df = None

    if args.supabase:
        try:
            logger.info("Fetching all job listings from Supabase to build corpus...")
            all_jobs_df = fetch_all_job_listings(supabase)
            if all_jobs_df.empty:
                logger.error("❌ No jobs found in the database. Aborting.")
                sys.exit(1)
            
            idf_corpus_df = all_jobs_df.copy()
            logger.info(f"Corpus built with {len(idf_corpus_df)} total jobs.")

            if args.job_id:
                logger.info(f"Fetching single job with ID: {args.job_id}")
                jobs_to_analyze_df = fetch_job_by_id(supabase, args.job_id)
                if jobs_to_analyze_df.empty:
                    logger.error(f"Could not find job with ID {args.job_id}. Aborting.")
                    sys.exit(1)

            else:
                unprocessed_jobs_df = all_jobs_df[all_jobs_df['processed_for_matching'] != True]

                if unprocessed_jobs_df.empty:
                    logger.info("✅ No new jobs to process. All caught up!")
                    return

                if args.analyze_all:
                    jobs_to_analyze_df = unprocessed_jobs_df
                    logger.info(f"📊 --analyze-all flag set. Analyzing all {len(jobs_to_analyze_df)} unprocessed jobs.")
                else:
                    jobs_to_analyze_df = unprocessed_jobs_df.head(args.batch_size)
                    logger.info(f"📊 Analyzing a batch of {len(jobs_to_analyze_df)} unprocessed jobs.")

        except Exception as e:
            logger.error(f"❌ Failed to fetch jobs from Supabase: {e}", exc_info=True)
            sys.exit(1)
                
    elif args.csv:
        if not os.path.exists(args.csv):
            logger.error(f"❌ CSV file not found: {args.csv}")
            sys.exit(1)
            
        try:
            jobs_df = load_and_process_jobs(args.csv)
            if jobs_df.empty:
                logger.error("❌ No valid jobs loaded from CSV")
                sys.exit(1)
            # In local mode, all jobs are for analysis and also form the corpus
            jobs_to_analyze_df = jobs_df
            idf_corpus_df = jobs_df.copy()
        except Exception as e:
            logger.error(f"❌ Failed to load CSV: {e}")
            sys.exit(1)

    # Run the pipeline
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
            logger.warning("⚠️ Pipeline completed, but no results were generated.")
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()