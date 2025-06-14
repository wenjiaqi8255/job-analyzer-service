import argparse
import logging
import os
from dotenv import load_dotenv
import pandas as pd
from supabase import create_client, Client
from analyzer.data_utils import load_and_process_jobs, fetch_jobs_to_analyze
from analyzer.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Job Analyzer Pipeline")
    parser.add_argument('--mode', choices=['train', 'inference'], default='inference', help='Pipeline mode')
    parser.add_argument('--csv', type=str, help='Path to CSV file for inference')
    parser.add_argument('--supabase', action='store_true', help='Use Supabase as data source and sink')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for Supabase inference')
    args = parser.parse_args()

    supabase = None
    jobs_df = None
    if args.supabase:
        load_dotenv()
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.error("SUPABASE_URL or SUPABASE_KEY not set in environment.")
            return
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        if args.mode == 'inference':
            jobs = fetch_jobs_to_analyze(supabase, batch_size=args.batch_size)
            if not jobs:
                logger.info("No jobs to process. Exiting.")
                return
            jobs_df = pd.DataFrame(jobs)
    elif args.csv:
        jobs_df = load_and_process_jobs(args.csv)
    else:
        logger.error("Please provide either --csv or --supabase.")
        return
    run_pipeline(args.mode, jobs_df, supabase=supabase)

if __name__ == "__main__":
    main() 