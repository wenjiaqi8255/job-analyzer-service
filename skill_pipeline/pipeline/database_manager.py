import hashlib
import json
import logging
from typing import Dict, Optional

from supabase import Client, create_client

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库操作管理器 - 分离语义数据和向量数据"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.supabase = self._init_supabase()

    def _init_supabase(self) -> Optional[Client]:
        """初始化Supabase客户端"""
        if not self.config.supabase_url or not self.config.supabase_key:
            logger.warning("⚠️ Supabase配置缺失，跳过数据库操作")
            return None

        try:
            client = create_client(self.config.supabase_url, self.config.supabase_key)
            logger.info("✅ Supabase客户端初始化成功")
            return client
        except Exception as e:
            logger.error(f"❌ Supabase客户端初始化失败: {e}")
            return None

    def upload_baselines(self, baselines: Dict) -> Dict:
        """上传基线数据到数据库（分离存储）"""
        if not self.supabase:
            logger.warning("⚠️ 数据库未连接，跳过上传")
            return {'status': 'skipped', 'reason': 'no_database_connection'}

        logger.info("📤 开始上传基线数据到分离的表...")
        results = {}

        # 上传全局基线
        results['global'] = self._upload_baseline_pair(
            'global', 'global', baselines['global']
        )

        # 上传角色基线
        results['roles'] = {}
        for role, baseline_data in baselines['roles'].items():
            results['roles'][role] = self._upload_baseline_pair(
                'role', role, baseline_data
            )

        # 上传行业基线
        results['industries'] = {}
        for industry, baseline_data in baselines['industries'].items():
            results['industries'][industry] = self._upload_baseline_pair(
                'industry', industry, baseline_data
            )

        logger.info("✅ 基线数据上传完成")
        return results

    def _upload_baseline_pair(self, baseline_type: str, name: str, baseline_data: Dict) -> Dict:
        """上传基线数据对（语义数据 + 向量数据）"""
        try:
            semantic_data = baseline_data.get('semantic_data', {})
            vector_data = baseline_data.get('vector_data', {})

            if not semantic_data:
                logger.warning(f"基线 {name} 缺少语义数据，跳过上传")
                return {'status': 'error', 'error': 'missing_semantic_data'}

            # 1. 首先上传语义数据到 semantic_baselines 表
            semantic_result = self._upload_semantic_baseline(baseline_type, name, semantic_data)

            if semantic_result['status'] != 'uploaded':
                logger.error(f"语义数据上传失败，跳过向量数据上传: {name}")
                return semantic_result

            baseline_id = semantic_result['id']

            # 2. 然后上传向量数据到 baseline_vectors 表
            if vector_data:
                vector_result = self._upload_vector_data(baseline_id, vector_data)

                return {
                    'status': 'uploaded',
                    'baseline_id': baseline_id,
                    'semantic_upload': semantic_result,
                    'vector_upload': vector_result
                }
            else:
                logger.warning(f"基线 {name} 缺少向量数据")
                return {
                    'status': 'partial',
                    'baseline_id': baseline_id,
                    'semantic_upload': semantic_result,
                    'vector_upload': {'status': 'skipped', 'reason': 'no_vector_data'}
                }

        except Exception as e:
            logger.error(f"上传基线对 {name} 失败: {e}")
            return {'status': 'error', 'error': str(e)}

    def _upload_semantic_baseline(self, baseline_type: str, name: str, semantic_data: Dict) -> Dict:
        """上传语义数据到 semantic_baselines 表"""
        source_hash = self._calculate_hash(semantic_data)

        # 检查是否已存在相同的语义数据
        try:
            result = self.supabase.table("semantic_baselines") \
                .select("*") \
                .eq("source_hash", source_hash) \
                .execute()

            if result.data:
                logger.info(f"语义基线 {name} 已存在，跳过上传")
                return {'status': 'exists', 'id': result.data[0]['id']}
        except Exception as e:
            logger.error(f"检查语义基线 {name} 时出错: {e}")
            return {'status': 'error', 'error': str(e)}

        # 插入新的语义数据
        insert_data = {
            "baseline_type": baseline_type,
            "name": name,
            "version": 1,
            "is_active": True,
            "baseline_data": semantic_data,  # 只包含人类可读的数据，不包含向量
            "source_hash": source_hash
        }

        try:
            result = self.supabase.table("semantic_baselines") \
                .insert(insert_data) \
                .execute()

            logger.info(f"✅ 成功上传语义基线: {name}")
            return {'status': 'uploaded', 'id': result.data[0]['id']}
        except Exception as e:
            logger.error(f"❌ 上传语义基线 {name} 失败: {e}")
            return {'status': 'error', 'error': str(e)}

    def _upload_vector_data(self, baseline_id: int, vector_data: Dict) -> Dict:
        """上传向量数据到 baseline_vectors 表"""
        # 检查是否已存在该baseline_id的向量数据
        try:
            result = self.supabase.table("baseline_vectors") \
                .select("*") \
                .eq("baseline_id", baseline_id) \
                .eq("embedding_model", "all-MiniLM-L6-v2") \
                .execute()

            if result.data:
                logger.info(f"向量数据 baseline_id={baseline_id} 已存在，跳过上传")
                return {'status': 'exists', 'id': result.data[0]['id']}
        except Exception as e:
            logger.error(f"检查向量数据 baseline_id={baseline_id} 时出错: {e}")
            return {'status': 'error', 'error': str(e)}

        # 插入新的向量数据
        insert_data = {
            "baseline_id": baseline_id,
            "embedding_model": "all-MiniLM-L6-v2",
            "vectors_data": vector_data  # 包含所有向量相关数据
        }

        try:
            result = self.supabase.table("baseline_vectors") \
                .insert(insert_data) \
                .execute()

            logger.info(f"✅ 成功上传向量数据: baseline_id={baseline_id}")
            return {'status': 'uploaded', 'id': result.data[0]['id']}
        except Exception as e:
            logger.error(f"❌ 上传向量数据 baseline_id={baseline_id} 失败: {e}")
            return {'status': 'error', 'error': str(e)}

    def _calculate_hash(self, data: dict) -> str:
        """计算数据哈希值"""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest() 