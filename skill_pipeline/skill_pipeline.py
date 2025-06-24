#!/usr/bin/env python3
"""
JobbAI 端到端技能处理管线
从vocabulary到三个baseline的完整流程
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from pipeline.config import PipelineConfig
from pipeline.orchestrator import PipelineOrchestrator


class EndToEndProcessor:
    """端到端处理器 - 主要接口类"""

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        初始化端到端处理器

        Args:
            config_path: 配置文件路径（可选）
            **kwargs: 配置参数覆盖
        """
        self.config = self._load_config(config_path, **kwargs)
        self.orchestrator = PipelineOrchestrator(self.config)

    def _load_config(self, config_path: Optional[str], **kwargs) -> PipelineConfig:
        """加载配置"""
        base_config = {}

        # 从配置文件加载
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                base_config = json.load(f)
            logger.info(f"✅ 从文件加载配置: {config_path}")

        # 应用参数覆盖
        base_config.update(kwargs)

        return PipelineConfig(**base_config)

    def process_vocabulary(self, vocab_path: str) -> Dict:
        """处理技能词汇表"""
        skills_vocab = self.orchestrator._load_vocabulary(vocab_path)
        all_skills = self.orchestrator.processor.extract_all_skills(skills_vocab)
        return {
            'vocabulary': skills_vocab,
            'skills': all_skills,
            'total_skills': len(all_skills)
        }

    def generate_classifications(self, all_skills: List[Dict]) -> Tuple[Dict, Dict]:
        """生成技能分类"""
        return self.orchestrator.classifier.classify_skills(all_skills)

    def generate_baselines(self, role_classifications: Dict, industry_classifications: Dict) -> Dict:
        """生成语义基线"""
        return self.orchestrator.baseline_generator.generate_all_baselines(
            role_classifications, industry_classifications
        )

    def upload_to_database(self, baselines: Dict) -> Dict:
        """上传基线到数据库"""
        return self.orchestrator.database_manager.upload_baselines(baselines)

    def run_complete_pipeline(self, vocab_path: str) -> Dict:
        """运行完整管线"""
        return self.orchestrator.run_complete_pipeline(vocab_path)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='JobbAI 端到端技能处理管线')

    # 输入输出参数
    parser.add_argument('vocabulary', help='技能词汇表文件路径')
    parser.add_argument('--output-dir', '-o', default='output', help='输出目录')
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--model-name', help='SentenceTransformer模型名称')

    # 分级阈值参数
    parser.add_argument('--high-confidence', type=float, default=0.75, help='高置信度阈值')
    parser.add_argument('--medium-confidence', type=float, default=0.55, help='中等置信度阈值')
    parser.add_argument('--low-confidence', type=float, default=0.35, help='低置信度阈值')

    # 兼容旧版本参数
    parser.add_argument('--role-threshold', type=float, help='角色分配阈值（兼容旧版本）')
    parser.add_argument('--industry-threshold', type=float, help='行业分配阈值（兼容旧版本）')

    # 调试选项
    parser.add_argument('--enable-debug', action='store_true', help='启用调试日志')
    parser.add_argument('--save-details', action='store_true', default=True, help='保存分类详情')

    # 数据库参数
    parser.add_argument('--supabase-url', help='Supabase URL')
    parser.add_argument('--supabase-key', help='Supabase API Key')
    parser.add_argument('--skip-database', action='store_true', help='跳过数据库上传')

    # 其他选项
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='日志级别')

    args = parser.parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # 准备配置参数
    config_kwargs = {
        'output_dir': args.output_dir
    }

    if args.model_name:
        config_kwargs['classification_model'] = args.model_name
    if args.high_confidence is not None:
        config_kwargs['high_confidence_threshold'] = args.high_confidence
    if args.medium_confidence is not None:
        config_kwargs['medium_confidence_threshold'] = args.medium_confidence
    if args.low_confidence is not None:
        config_kwargs['low_confidence_threshold'] = args.low_confidence

    # 兼容旧版本参数
    if args.role_threshold is not None:
        config_kwargs['role_threshold'] = args.role_threshold
    if args.industry_threshold is not None:
        config_kwargs['industry_threshold'] = args.industry_threshold

    if args.supabase_url:
        config_kwargs['supabase_url'] = args.supabase_url
    if args.supabase_key:
        config_kwargs['supabase_key'] = args.supabase_key
    if args.skip_database:
        config_kwargs['supabase_url'] = None
        config_kwargs['supabase_key'] = None
    if args.enable_debug:
        config_kwargs['enable_debug_logging'] = True
    if args.save_details is not None:
        config_kwargs['save_classification_details'] = args.save_details

    print("🚀 JobbAI 端到端技能处理管线")
    print(f"📁 词汇表文件: {args.vocabulary}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🔧 配置文件: {args.config or '使用默认配置'}")
    print("-" * 60)

    try:
        # 初始化处理器
        processor = EndToEndProcessor(args.config, **config_kwargs)

        # 运行完整管线
        results = processor.run_complete_pipeline(args.vocabulary)

        print("\n🎉 管线执行成功完成!")
        return 0

    except Exception as e:
        logger.error(f"❌ 管线执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())