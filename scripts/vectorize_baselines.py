import os
import sys
import logging
from dotenv import load_dotenv
import torch
import json

# Add project root to path to allow importing from other directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Database Connection ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logging.error("Supabase URL and Key must be set in .env file.")
    sys.exit(1)

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logging.info("✅ Successfully connected to Supabase.")
except Exception as e:
    logging.error(f"❌ Failed to connect to Supabase: {e}")
    sys.exit(1)

# --- Model Loading ---
try:
    logging.info(f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load embedding model: {e}")
    sys.exit(1)

def vectorize_and_store_baselines():
    """
    Fetches all semantic baselines, generates vector embeddings for their
    keywords, and upserts them into the baseline_vectors table.
    """
    logging.info("Fetching all semantic baselines from the database...")
    
    # 1. Fetch all baselines (not just active ones)
    response = supabase.table('semantic_baselines').select('id, name, baseline_data').execute()
    
    if not response.data:
        logging.warning("No semantic baselines found to process.")
        return

    baselines = response.data
    logging.info(f"Found {len(baselines)} baselines to process.")

    processed_count = 0
    for baseline in baselines:
        baseline_id = baseline['id']
        baseline_name = baseline['name']
        baseline_data = baseline.get('baseline_data', {})

        # Updated logic to handle nested keyword structure
        keywords = []
        if isinstance(baseline_data, dict):
            for category_keywords in baseline_data.values():
                if isinstance(category_keywords, list):
                    keywords.extend(category_keywords)
        
        if not keywords:
            logging.warning(f"Skipping baseline '{baseline_name}' (ID: {baseline_id}) as it has no keywords.")
            continue

        logging.info(f"Processing baseline '{baseline_name}' (ID: {baseline_id}) with {len(keywords)} keywords...")

        try:
            # 2. Generate embeddings
            embeddings = model.encode(keywords)
            
            # 3. Structure for JSON storage
            vectors_data = {keyword: vector.tolist() for keyword, vector in zip(keywords, embeddings)}

            # 4. Prepare data for upsert
            data_to_upsert = {
                'baseline_id': baseline_id,
                'embedding_model': EMBEDDING_MODEL_NAME,
                'vectors_data': json.dumps(vectors_data) # Ensure it's a JSON string if the column expects it
            }

            # 5. Upsert into the database. This will insert a new row or update an
            # existing one if a row with the same baseline_id and embedding_model already exists.
            # NOTE: This requires a UNIQUE constraint on (baseline_id, embedding_model) in the database.
            supabase.table('baseline_vectors').upsert(
                data_to_upsert
            ).execute()

            processed_count += 1
            logging.info(f"✅ Successfully vectorized and stored baseline '{baseline_name}'.")

        except Exception as e:
            logging.error(f"❌ Failed to process baseline '{baseline_name}' (ID: {baseline_id}): {e}", exc_info=True)

    logging.info(f"\n--- Process Complete ---")
    logging.info(f"Successfully processed and stored vectors for {processed_count}/{len(baselines)} baselines.")

def main():
    print("This script will vectorize all keywords in the 'semantic_baselines' table")
    print(f"and store them in the 'baseline_vectors' table using the '{EMBEDDING_MODEL_NAME}' model.")
    print("This will overwrite existing vectors for this model if they already exist.")
    print("\n\nIMPORTANT:")
    print("To allow the script to safely update existing entries, you should add a unique constraint to your table.")
    print("You can do this by running the following SQL command in your database:")
    print("\nALTER TABLE public.baseline_vectors ADD CONSTRAINT unique_baseline_model UNIQUE (baseline_id, embedding_model);\n")
    
    user_input = input("Do you want to continue? (y/n): ")
    if user_input.lower() == 'y':
        vectorize_and_store_baselines()
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main() 