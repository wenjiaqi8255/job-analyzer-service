import argparse
import logging
import os
import sys
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
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for Supabase inference')  # 调整默认值
    args = parser.parse_args()

    supabase = None
    jobs_df = None
    
    if args.supabase:
        # GitHub Actions环境下不需要load_dotenv()
        if os.path.exists('.env'):
            load_dotenv()
            
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.error("SUPABASE_URL or SUPABASE_KEY not set in environment.")
            sys.exit(1)  # 使用sys.exit而不是return
            
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("✅ Successfully connected to Supabase")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            sys.exit(1)
            
        if args.mode == 'inference':
            try:
                jobs = fetch_jobs_to_analyze(supabase, batch_size=args.batch_size)
                if not jobs:
                    logger.info("✅ No new jobs to process. All caught up!")
                    return  # 这种情况是正常的，不是错误
                    
                jobs_df = pd.DataFrame(jobs)
                logger.info(f"📊 Loaded {len(jobs_df)} jobs for analysis")
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch jobs from Supabase: {e}")
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
        except Exception as e:
            logger.error(f"❌ Failed to load CSV: {e}")
            sys.exit(1)
    else:
        logger.error("❌ Please provide either --csv or --supabase.")
        sys.exit(1)

    # 运行pipeline
    try:
        logger.info(f"🚀 Starting {args.mode} pipeline...")
        results = run_pipeline(args.mode, jobs_df, supabase=supabase)
        
        if results:
            logger.info(f"✅ Pipeline completed successfully! Processed {len(results)} jobs.")
        else:
            logger.warning("⚠️ Pipeline completed but no results generated.")
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()