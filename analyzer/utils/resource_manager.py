import logging
import torch
import spacy
from sentence_transformers import SentenceTransformer
from ..config import AppConfig
from .exceptions import ModelLoadingError
from .db_queries import fetch_active_semantic_baselines, fetch_baseline_vectors
from supabase import Client

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    A context manager to handle the lifecycle of heavy resources like models and database connections.
    """
    def __init__(self, config: AppConfig, supabase_client: Client = None):
        self.config = config
        self.supabase_client = supabase_client
        self.embedding_model = None
        self.nlp_model = None
        self.semantic_baselines = {"role": {}, "industry": {}, "global": {}}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            self.device = 'mps'


    def __enter__(self):
        """
        Loads models and baselines.
        """
        self._load_embedding_model()
        self._load_spacy_model()
        self._load_baselines_from_db()
        
        return {
            "embedding_model": self.embedding_model,
            "nlp_model": self.nlp_model,
            "semantic_baselines": self.semantic_baselines
        }

    def _load_embedding_model(self):
        logger.info(f"Loading embedding model '{self.config.model_name}'...")
        try:
            self.embedding_model = SentenceTransformer(self.config.model_name)
            if torch.cuda.is_available():
                self.embedding_model.to(torch.device("cuda"))
                logger.info("Model moved to GPU.")
        except Exception as e:
            raise ModelLoadingError(f"Failed to load SentenceTransformer model '{self.config.model_name}': {e}") from e

    def _load_spacy_model(self):
        logger.info(f"Loading spaCy model '{self.config.spacy_model_name}'...")
        try:
            self.nlp_model = spacy.load(self.config.spacy_model_name)
        except OSError:
            logger.warning(f"spaCy model '{self.config.spacy_model_name}' not found. Downloading...")
            spacy.cli.download(self.config.spacy_model_name)
            self.nlp_model = spacy.load(self.config.spacy_model_name)
        except Exception as e:
            raise ModelLoadingError(f"Failed to load spaCy model: {e}") from e

    def _load_baselines_from_db(self):
        if not self.supabase_client:
            logger.warning("Supabase client not provided. Skipping loading of semantic baselines.")
            return
            
        logger.info("Loading semantic baselines from database...")
        baselines = fetch_active_semantic_baselines(self.supabase_client)
        if not baselines:
            logger.warning("No baselines loaded from DB.")
            self.semantic_baselines = {"role": {}, "industry": {}, "global": {}}
            return

        logger.debug(f"Fetched {len(baselines)} baseline records from DB.")

        for i, b in enumerate(baselines):
            logger.debug(f"Processing baseline record {i+1}/{len(baselines)}: {{'id': {b.get('id')}, 'name': '{b.get('name')}, 'type': '{b.get('baseline_type')}'}}")
            baseline_type = b.get('baseline_type')
            name = b.get('name')
            baseline_id = b.get('id')

            if not (baseline_type and name and baseline_id):
                logger.warning(f"Skipping baseline record due to missing fields: {b}")
                continue
            
            vector_data = fetch_baseline_vectors(self.supabase_client, baseline_id, self.config.model_name)
            logger.debug(f"For baseline '{name}', vector_data fetched from DB: {vector_data is not None}")
            
            vectors = None
            if vector_data and 'vectors_data' in vector_data[0]:
                vectors_payload = vector_data[0]['vectors_data']
                vectors_list = []

                # The actual vector data is nested inside the JSON payload
                if isinstance(vectors_payload, dict) and 'vectors' in vectors_payload:
                    vectors_list = vectors_payload.get('vectors')
                elif isinstance(vectors_payload, list): # Keep backward compatibility
                    vectors_list = vectors_payload

                if vectors_list:
                    try:
                        vectors = torch.tensor(vectors_list, device=self.device)
                        logger.debug(f"Successfully created tensor for '{name}' with shape {vectors.shape}")
                    except Exception as e:
                        logger.error(f"Could not convert vectors to tensor for '{name}': {e}", exc_info=True)
                else:
                    logger.warning(f"Vector list for '{name}' is empty after extraction. Skipping tensor creation.")
            else:
                logger.warning(f"No valid 'vectors_data' key in vector_data for '{name}'.")
            
            # A baseline is only useful if it has vectors.
            if vectors is not None:
                # Correctly extract keywords from the nested 'baseline_data' field
                keywords = b.get('baseline_data', {}).get('skills_list', [])
                if baseline_type in self.semantic_baselines:
                    self.semantic_baselines[baseline_type][name] = {
                        "vectors": vectors,
                        "keywords": keywords
                    }
                    logger.debug(f"Successfully added baseline '{name}' of type '{baseline_type}'.")
                else:
                    logger.warning(f"Baseline type '{baseline_type}' is not a recognized type. Skipping.")
            else:
                logger.warning(f"No vectors were created for baseline '{name}', so it will not be added.")
        
        logger.info("Semantic baselines loading process complete.")
        logger.debug(f"Final semantic_baselines keys: {list(self.semantic_baselines.keys())}")
        logger.debug(f"Number of 'role' baselines: {len(self.semantic_baselines.get('role', {}))}")
        logger.debug(f"Number of 'industry' baselines: {len(self.semantic_baselines.get('industry', {}))}")
        logger.debug(f"Number of 'global' baselines: {len(self.semantic_baselines.get('global', {}))}")


    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Cleans up resources.
        """
        logger.info("Releasing resources...")
        if self.embedding_model is not None:
            if hasattr(self.embedding_model, 'cpu'):
                self.embedding_model.cpu()
            del self.embedding_model
            self.embedding_model = None

        if self.nlp_model is not None:
            del self.nlp_model
            self.nlp_model = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.") 