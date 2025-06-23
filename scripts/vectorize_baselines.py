# import os
# import sys
# import logging
# from dotenv import load_dotenv
# import torch
# import json

# # Add project root to path to allow importing from other directories
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from supabase import create_client, Client
# from sentence_transformers import SentenceTransformer

# # --- Setup ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# load_dotenv()

# # --- Configuration - 使用与 skill_pipeline.py 相同的模型配置 ---
# PRIMARY_MODEL_NAME = 'all-MiniLM-L6-v2'
# BACKUP_MODEL_NAME = 'all-mpnet-base-v2'

# # --- Database Connection ---
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# if not SUPABASE_URL or not SUPABASE_KEY:
#     logging.error("Supabase URL and Key must be set in .env file.")
#     sys.exit(1)

# try:
#     supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
#     logging.info("✅ Successfully connected to Supabase.")
# except Exception as e:
#     logging.error(f"❌ Failed to connect to Supabase: {e}")
#     sys.exit(1)

# # --- Model Loading - 与 skill_pipeline.py 保持一致的加载逻辑 ---
# def load_model():
#     """加载与 skill_pipeline.py 相同的模型"""
#     logging.info(f"🤖 尝试加载主模型: {PRIMARY_MODEL_NAME}...")
#     try:
#         model = SentenceTransformer(PRIMARY_MODEL_NAME)
#         logging.info(f"✅ 成功加载专用模型: {PRIMARY_MODEL_NAME}")
#         return model, PRIMARY_MODEL_NAME
#     except Exception as e:
#         logging.warning(f"⚠️ 无法加载专用模型，使用备选模型: {e}")
#         try:
#             model = SentenceTransformer(BACKUP_MODEL_NAME)
#             logging.info(f"✅ 成功加载备选模型: {BACKUP_MODEL_NAME}")
#             return model, BACKUP_MODEL_NAME
#         except Exception as e2:
#             logging.error(f"❌ 无法加载任何模型: {e2}")
#             sys.exit(1)

# model, actual_model_name = load_model()

# def vectorize_and_store_baselines():
#     """
#     Fetches all semantic baselines, generates vector embeddings for their
#     keywords using the SAME model as skill_pipeline.py, and upserts them 
#     into the baseline_vectors table.
#     """
#     logging.info("Fetching all semantic baselines from the database...")
    
#     # 1. Fetch all baselines (not just active ones)
#     response = supabase.table('semantic_baselines').select('id, name, baseline_data').execute()
    
#     if not response.data:
#         logging.warning("No semantic baselines found to process.")
#         return

#     baselines = response.data
#     logging.info(f"Found {len(baselines)} baselines to process.")

#     processed_count = 0
#     for baseline in baselines:
#         baseline_id = baseline['id']
#         baseline_name = baseline['name']
#         baseline_data = baseline.get('baseline_data', {})

#         # Updated logic to handle nested keyword structure
#         keywords = []
#         if isinstance(baseline_data, dict):
#             # 处理三种可能的数据结构
#             if 'categories' in baseline_data:
#                 # 角色/行业基线结构：有 categories 字段
#                 categories = baseline_data['categories']
#                 for category_keywords in categories.values():
#                     if isinstance(category_keywords, list):
#                         keywords.extend(category_keywords)
#             elif any(isinstance(v, dict) for v in baseline_data.values()):
#                 # 嵌套字典结构
#                 for category_data in baseline_data.values():
#                     if isinstance(category_data, dict):
#                         for category_keywords in category_data.values():
#                             if isinstance(category_keywords, list):
#                                 keywords.extend(category_keywords)
#             else:
#                 # 简单字典结构
#                 for category_keywords in baseline_data.values():
#                     if isinstance(category_keywords, list):
#                         keywords.extend(category_keywords)
        
#         if not keywords:
#             logging.warning(f"Skipping baseline '{baseline_name}' (ID: {baseline_id}) as it has no keywords.")
#             continue

#         # 确保所有关键词都是字符串，过滤None，然后去重并排序
#         keywords = sorted(list(set(str(k) for k in keywords if k is not None)))
#         logging.info(f"Processing baseline '{baseline_name}' (ID: {baseline_id}) with {len(keywords)} unique keywords...")

#         try:
#             # 2. Generate embeddings using the same model as skill_pipeline.py
#             embeddings = model.encode(keywords)
            
#             # 3. Structure for JSON storage
#             vectors_data = {keyword: vector.tolist() for keyword, vector in zip(keywords, embeddings)}

#             # 4. Prepare data for upsert - 使用实际加载的模型名称
#             data_to_upsert = {
#                 'baseline_id': baseline_id,
#                 'embedding_model': actual_model_name,  # 使用实际加载的模型名称
#                 'vectors_data': json.dumps(vectors_data)
#             }

#             # 5. Upsert into the database
#             supabase.table('baseline_vectors').upsert(
#                 data_to_upsert
#             ).execute()

#             processed_count += 1
#             logging.info(f"✅ Successfully vectorized and stored baseline '{baseline_name}' using {actual_model_name}.")

#         except Exception as e:
#             logging.error(f"❌ Failed to process baseline '{baseline_name}' (ID: {baseline_id}): {e}", exc_info=True)

#     logging.info(f"\n--- Process Complete ---")
#     logging.info(f"Successfully processed and stored vectors for {processed_count}/{len(baselines)} baselines.")
#     logging.info(f"All vectors generated using model: {actual_model_name}")

# def check_model_consistency():
#     """检查数据库中是否存在使用不同模型的向量"""
#     logging.info("🔍 检查模型一致性...")
    
#     try:
#         response = supabase.table('baseline_vectors').select('embedding_model').execute()
        
#         if response.data:
#             existing_models = set(row['embedding_model'] for row in response.data)
#             logging.info(f"数据库中现有的模型: {existing_models}")
            
#             if len(existing_models) > 1:
#                 logging.warning(f"⚠️ 检测到多个不同的模型！这可能导致向量不兼容：{existing_models}")
#                 logging.warning("建议删除旧的向量数据或确保所有系统使用相同模型")
#             elif existing_models and actual_model_name not in existing_models:
#                 logging.warning(f"⚠️ 当前模型 {actual_model_name} 与数据库中的模型不匹配：{existing_models}")
#                 logging.warning("这将导致向量空间不兼容！")
#             else:
#                 logging.info("✅ 模型一致性检查通过")
#         else:
#             logging.info("数据库中暂无向量数据")
#     except Exception as e:
#         logging.error(f"模型一致性检查失败: {e}")

# def main():
#     print("JobbAI Baseline Vectorization Tool")
#     print("=" * 50)
#     print(f"🤖 使用模型: {actual_model_name}")
#     print("此脚本将使用与 skill_pipeline.py 相同的模型对语义基线进行向量化")
#     print("并将结果存储到 'baseline_vectors' 表中")
    
#     # 检查模型一致性
#     check_model_consistency()
    
#     print(f"\n⚠️  重要说明:")
#     print("为了确保向量兼容性，请确保:")
#     print("1. skill_pipeline.py 和此脚本使用相同的模型")
#     print("2. 如果更换模型，需要重新生成所有向量")
#     print("3. 建议在数据库中添加唯一约束:")
#     print("   ALTER TABLE public.baseline_vectors ADD CONSTRAINT unique_baseline_model UNIQUE (baseline_id, embedding_model);")
    
#     user_input = input(f"\n继续使用 {actual_model_name} 进行向量化? (y/n): ")
#     if user_input.lower() == 'y':
#         vectorize_and_store_baselines()
#     else:
#         print("操作已取消。")

# if __name__ == "__main__":
#     main()