# utils/job_data_accessor.py (新文件)
from typing import Dict

class JobDataAccessor:
    """统一的Job数据访问工具"""
    
    @staticmethod
    def get_effective_description(job_data: Dict) -> str:
        """获取用于分析的有效描述"""
        translated = job_data.get('translated_description')
        if translated and translated.strip():
            return translated
        return job_data.get('description', '')
    
    @staticmethod
    def get_description_metadata(job_data: Dict) -> Dict:
        """获取描述的元数据信息"""
        has_translation = bool(job_data.get('translated_description'))
        return {
            'has_translation': has_translation,
            'effective_language': 'en' if has_translation else 'unknown',
            'description_source': 'translated' if has_translation else 'original'
        }
    
    @staticmethod
    def prepare_job_for_analysis(job_data: Dict) -> Dict:
        """为分析准备job数据，添加有效字段"""
        enriched_job = job_data.copy()
        enriched_job['effective_description'] = JobDataAccessor.get_effective_description(job_data)
        enriched_job.update(JobDataAccessor.get_description_metadata(job_data))
        return enriched_job