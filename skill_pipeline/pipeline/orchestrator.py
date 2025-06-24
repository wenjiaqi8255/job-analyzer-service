import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .config import PipelineConfig
from .database_manager import DatabaseManager
from .skill_processor import SkillProcessor
from .semantic_classifier import SemanticClassifier
from .baseline_generator import BaselineGenerator

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """整体流程编排器"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.processor = SkillProcessor()
        config_dir = config.semantic_config_dir  # 新增配置目录路径
        self.classifier = SemanticClassifier(config_dir)
        # 修改初始化基线生成器的调用
        self.baseline_generator = BaselineGenerator(config, self.classifier.semantic_engine.model)
        self.database_manager = DatabaseManager(config)

        # 创建输出目录
        Path(self.config.output_dir).mkdir(exist_ok=True)

    def run_complete_pipeline(self, vocab_path: str) -> Dict:
        """运行完整管线"""
        logger.info("🚀 启动端到端技能处理管线")
        logger.info(f"📁 输入文件: {vocab_path}")
        logger.info(f"📁 输出目录: {self.config.output_dir}")
        print("-" * 60)

        pipeline_start = datetime.now()
        results = {}

        try:
            # 1. 加载和处理词汇表
            logger.info("📊 Step 1: 加载技能词汇表...")
            skills_vocab = self._load_vocabulary(vocab_path)
            all_skills = self.processor.extract_all_skills(skills_vocab)
            logger.info(f"📋 提取到 {len(all_skills)} 个技能")
            results['skills_extracted'] = len(all_skills)

            # 2. 语义分类
            logger.info("🧠 Step 2: 执行语义分类...")
            role_classifications, industry_classifications = self.classifier.classify_skills(all_skills)
            results['role_classifications'] = role_classifications
            results['industry_classifications'] = industry_classifications

            # 保存分类结果
            self._save_classifications(role_classifications, industry_classifications)

            # 3. 生成语义基线
            logger.info("🏗️ Step 3: 生成语义基线...")
            baselines = self.baseline_generator.generate_all_baselines(
                role_classifications, industry_classifications
            )
            results['baselines'] = baselines

            # 保存基线文件
            self._save_baselines(baselines)

            # 4. 上传到数据库
            logger.info("📤 Step 4: 上传到数据库...")
            upload_results = self.database_manager.upload_baselines(baselines)
            results['upload_results'] = upload_results

            # 5. 生成总结报告
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()

            results['summary'] = self._generate_summary_report(
                all_skills, role_classifications, industry_classifications,
                baselines, upload_results, duration
            )

            # 保存总结报告
            self._save_summary_report(results['summary'])

            logger.info("🎉 管线执行完成!")
            self._print_summary(results['summary'])

        except Exception as e:
            logger.error(f"❌ 管线执行失败: {e}")
            results['error'] = str(e)
            raise

        return results

    def _load_vocabulary(self, vocab_path: str) -> Dict:
        """加载技能词汇表"""
        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(f"词汇表文件不存在: {vocab_path}")

        with open(vocab_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 尝试直接解析JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 如果失败，尝试从markdown中提取JSON
            import re
            json_match = re.search(r'\{[\s\S]*?\}(?=\s*###|\s*$)', content)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("无法解析技能词汇表文件")

    def _save_classifications(self, role_classifications: Dict, industry_classifications: Dict):
        """保存分类结果"""
        # 保存角色分类
        role_path = Path(self.config.output_dir) / "role_skill_classifications.json"
        with open(role_path, 'w', encoding='utf-8') as f:
            json.dump(role_classifications, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 角色分类已保存: {role_path}")

        # 保存行业分类
        industry_path = Path(self.config.output_dir) / "industry_skill_classifications.json"
        with open(industry_path, 'w', encoding='utf-8') as f:
            json.dump(industry_classifications, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 行业分类已保存: {industry_path}")

        # 保存分类详情（如果启用调试）
        if self.config.save_classification_details:
            self._save_classification_details(role_classifications, industry_classifications)

    def _save_classification_details(self, role_classifications: Dict, industry_classifications: Dict):
        """保存分类详情统计"""
        details = {
            'classification_summary': {
                'total_roles_with_skills': len(
                    [r for r in role_classifications.values() if any(skills for skills in r.values())]),
                'total_industries_with_skills': len(
                    [i for i in industry_classifications.values() if any(skills for skills in i.values())]),
                'role_skill_distribution': {
                    role: {cat: len(skills) for cat, skills in categories.items() if skills}
                    for role, categories in role_classifications.items()
                    if any(skills for skills in categories.values())
                },
                'industry_skill_distribution': {
                    industry: {cat: len(skills) for cat, skills in categories.items() if skills}
                    for industry, categories in industry_classifications.items()
                    if any(skills for skills in categories.values())
                }
            },
            'quality_insights': {
                'largest_role_categories': self._get_largest_categories(role_classifications),
                'largest_industry_categories': self._get_largest_categories(industry_classifications),
                'potential_issues': self._identify_potential_issues(role_classifications, industry_classifications)
            },
            'configuration_used': {
                'high_confidence_threshold': self.config.high_confidence_threshold,
                'medium_confidence_threshold': self.config.medium_confidence_threshold,
                'low_confidence_threshold': self.config.low_confidence_threshold,
                'model_name': self.config.classification_model
            }
        }

        details_path = Path(self.config.output_dir) / "classification_details.json"
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        logger.info(f"📋 分类详情已保存: {details_path}")

    def _get_largest_categories(self, classifications: Dict) -> List[Dict]:
        """获取最大的分类类别"""
        largest = []
        for name, categories in classifications.items():
            for category, skills in categories.items():
                if skills:
                    largest.append({
                        'name': name,
                        'category': category,
                        'skill_count': len(skills),
                        'sample_skills': skills[:5]  # 只保存前5个技能作为示例
                    })

        # 按技能数量排序
        largest.sort(key=lambda x: x['skill_count'], reverse=True)
        return largest[:10]  # 返回前10个最大的类别

    def _identify_potential_issues(self, role_classifications: Dict, industry_classifications: Dict) -> List[str]:
        """识别潜在的分类问题"""
        issues = []

        # 检查是否有角色占用过多技能
        total_role_skills = sum(sum(len(skills) for skills in categories.values())
                                for categories in role_classifications.values())

        for role, categories in role_classifications.items():
            role_skills = sum(len(skills) for skills in categories.values())
            if total_role_skills > 0 and role_skills > total_role_skills * 0.3:  # 如果单个角色占用超过30%的技能
                issues.append(
                    f"角色 '{role}' 可能占用了过多技能 ({role_skills} 个, {role_skills / total_role_skills * 100:.1f}%)")

        # 检查是否有行业为空
        empty_industries = [industry for industry, categories in industry_classifications.items()
                            if not any(skills for skills in categories.values())]
        if len(empty_industries) > len(industry_classifications) * 0.5:
            issues.append(f"超过一半的行业分类为空 ({len(empty_industries)}/{len(industry_classifications)})")

        # 检查特定技能是否被正确分类
        design_tools = ['figma', 'sketch', 'adobe_xd']
        ui_ux_skills = []
        if 'ui_ux_designer' in role_classifications:
            ui_ux_skills = [skill for category in role_classifications['ui_ux_designer'].values()
                            for skill in category]

        missing_design_tools = [tool for tool in design_tools if tool not in ui_ux_skills]
        if missing_design_tools:
            issues.append(f"设计工具可能未正确分类到UI/UX设计师: {missing_design_tools}")

        return issues

    def _save_baselines(self, baselines: Dict):
        """保存基线文件（分离语义和向量数据）"""
        # 保存全局基线 - 语义数据
        global_semantic_path = Path(self.config.output_dir) / "global_baseline_semantic.json"
        with open(global_semantic_path, 'w', encoding='utf-8') as f:
            json.dump(baselines['global']['semantic_data'], f, indent=2, ensure_ascii=False)
        logger.info(f"💾 全局语义基线已保存: {global_semantic_path}")

        # 保存全局基线 - 向量数据
        global_vector_path = Path(self.config.output_dir) / "global_baseline_vectors.json"
        with open(global_vector_path, 'w', encoding='utf-8') as f:
            json.dump(baselines['global']['vector_data'], f, indent=2, ensure_ascii=False)
        logger.info(f"💾 全局向量基线已保存: {global_vector_path}")

        # 保存角色基线 - 语义数据
        role_semantic_data = {role: data['semantic_data'] for role, data in baselines['roles'].items()}
        role_semantic_path = Path(self.config.output_dir) / "role_baselines_semantic.json"
        with open(role_semantic_path, 'w', encoding='utf-8') as f:
            json.dump(role_semantic_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 角色语义基线已保存: {role_semantic_path}")

        # 保存角色基线 - 向量数据
        role_vector_data = {role: data['vector_data'] for role, data in baselines['roles'].items()}
        role_vector_path = Path(self.config.output_dir) / "role_baselines_vectors.json"
        with open(role_vector_path, 'w', encoding='utf-8') as f:
            json.dump(role_vector_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 角色向量基线已保存: {role_vector_path}")

        # 保存行业基线 - 语义数据
        industry_semantic_data = {industry: data['semantic_data'] for industry, data in
                                  baselines['industries'].items()}
        industry_semantic_path = Path(self.config.output_dir) / "industry_baselines_semantic.json"
        with open(industry_semantic_path, 'w', encoding='utf-8') as f:
            json.dump(industry_semantic_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 行业语义基线已保存: {industry_semantic_path}")

        # 保存行业基线 - 向量数据
        industry_vector_data = {industry: data['vector_data'] for industry, data in baselines['industries'].items()}
        industry_vector_path = Path(self.config.output_dir) / "industry_baselines_vectors.json"
        with open(industry_vector_path, 'w', encoding='utf-8') as f:
            json.dump(industry_vector_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 行业向量基线已保存: {industry_vector_path}")

    def _count_semantic_uploads(self, upload_results: Dict) -> int:
        """统计语义数据上传数量"""
        count = 0
        if upload_results.get('global', {}).get('semantic_upload', {}).get('status') in ['uploaded', 'exists']:
            count += 1
        for result in upload_results.get('roles', {}).values():
            if result.get('semantic_upload', {}).get('status') in ['uploaded', 'exists']:
                count += 1
        for result in upload_results.get('industries', {}).values():
            if result.get('semantic_upload', {}).get('status') in ['uploaded', 'exists']:
                count += 1
        return count

    def _count_vector_uploads(self, upload_results: Dict) -> int:
        """统计成功上传的向量数量"""
        count = 0
        if 'global' in upload_results and upload_results['global'].get('vector_data', {}).get('success'):
            count += 1
        if 'roles' in upload_results:
            count += sum(1 for res in upload_results['roles'].values() if res.get('vector_data', {}).get('success'))
        if 'industries' in upload_results:
            count += sum(
                1 for res in upload_results['industries'].values() if res.get('vector_data', {}).get('success'))
        return count

    def _generate_summary_report(self, all_skills: List[Dict], role_classifications: Dict,
                                industry_classifications: Dict, baselines: Dict,
                                upload_results: Dict, duration: float) -> Dict:
        """生成总结报告"""
        # 增加健壮性，处理 upload_results 可能不是字典的情况
        if not isinstance(upload_results, dict):
            logger.warning(f"⚠️ 'upload_results' 不是一个字典, 而是 {type(upload_results)}. "
                           f"将使用空的上传统计数据. 收到的值: {upload_results}")
            upload_results = {}
            
        total_skills = len(all_skills)
        total_roles = len(role_classifications)
        total_industries = len(industry_classifications)

        # 统计分类结果
        total_role_skills = sum(sum(len(skills) for skills in categories.values())
                                for categories in role_classifications.values())
        total_industry_skills = sum(sum(len(skills) for skills in categories.values())
                                    for categories in industry_classifications.values())

        # 统计上传结果 - 更新以支持新的分离上传格式
        upload_stats = {
            'global_uploaded': upload_results.get('global', {}).get('status') == 'uploaded',
            'roles_uploaded': sum(1 for result in upload_results.get('roles', {}).values()
                                if result.get('status') == 'uploaded'),
            'industries_uploaded': sum(1 for result in upload_results.get('industries', {}).values()
                                       if result.get('status') == 'uploaded'),
            'total_semantic_uploads': self._count_semantic_uploads(upload_results),
            'total_vector_uploads': self._count_vector_uploads(upload_results),
            'total_errors': sum(1 for section in upload_results.values()
                              for result in (section.values() if isinstance(section, dict) else [section])
                              if result.get('status') == 'error')
        }

        return {
            'pipeline_info': {
                'execution_time_seconds': round(duration, 2),
                'completion_time': datetime.now().isoformat(),
                'config': {
                    'model_name': self.config.classification_model,
                    'role_threshold': self.config.role_threshold,
                    'industry_threshold': self.config.industry_threshold,
                    'min_skills_for_baseline': self.config.min_skills_for_baseline
                }
            },
            'skill_processing': {
                'total_skills_extracted': len(all_skills),
                'total_role_assignments': total_role_skills,
                'total_industry_assignments': total_industry_skills,
                'roles_with_skills': len(role_classifications),
                'industries_with_skills': len(industry_classifications)
            },
            'baseline_generation': {
                'global_baseline_generated': 'global' in baselines,
                'role_baselines_generated': len(baselines.get('roles', {})),
                'industry_baselines_generated': len(baselines.get('industries', {})),
                'total_baselines': 1 + len(baselines.get('roles', {})) + len(baselines.get('industries', {}))
            },
            'database_upload': upload_stats,
            'quality_metrics': {
                'classification_coverage': {
                    'role_coverage_percent': round(
                        (total_role_skills / len(all_skills)) * 100, 2) if all_skills else 0,
                    'industry_coverage_percent': round(
                        (total_industry_skills / len(all_skills)) * 100, 2) if all_skills else 0
                },
                'baseline_quality': {
                    'avg_skills_per_role': round(total_role_skills / len(
                        role_classifications), 2) if role_classifications else 0,
                    'avg_skills_per_industry': round(total_industry_skills / len(
                        industry_classifications), 2) if industry_classifications else 0
                }
            },
            'output_files': {
                'role_classifications': f"{self.config.output_dir}/role_skill_classifications.json",
                'industry_classifications': f"{self.config.output_dir}/industry_skill_classifications.json",
                # 新的分离格式文件
                'global_baseline_semantic': f"{self.config.output_dir}/global_baseline_semantic.json",
                'global_baseline_vectors': f"{self.config.output_dir}/global_baseline_vectors.json",
                'role_baselines_semantic': f"{self.config.output_dir}/role_baselines_semantic.json",
                'role_baselines_vectors': f"{self.config.output_dir}/role_baselines_vectors.json",
                'industry_baselines_semantic': f"{self.config.output_dir}/industry_baselines_semantic.json",
                'industry_baselines_vectors': f"{self.config.output_dir}/industry_baselines_vectors.json",
                'summary_report': f"{self.config.output_dir}/pipeline_summary.json"
            }
        }

    def _save_summary_report(self, summary: Dict):
        """保存总结报告"""
        summary_path = Path(self.config.output_dir) / "pipeline_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 总结报告已保存: {summary_path}")

    def _print_summary(self, summary: Dict):
        """打印总结信息"""
        print("\n" + "=" * 60)
        print("📊 管线执行总结")
        print("=" * 60)

        # 基本信息
        pipeline_info = summary['pipeline_info']
        print(f"⏱️  执行时间: {pipeline_info['execution_time_seconds']} 秒")
        print(f"🕐 完成时间: {pipeline_info['completion_time']}")

        # 技能处理
        skill_info = summary['skill_processing']
        print(f"\n📋 技能处理:")
        print(f"   总技能数: {skill_info['total_skills_extracted']}")
        print(f"   角色分配: {skill_info['total_role_assignments']}")
        print(f"   行业分配: {skill_info['total_industry_assignments']}")
        print(f"   有效角色: {skill_info['roles_with_skills']}")
        print(f"   有效行业: {skill_info['industries_with_skills']}")

        # 基线生成
        baseline_info = summary['baseline_generation']
        print(f"\n🏗️ 基线生成:")
        print(f"   总基线数: {baseline_info['total_baselines']}")
        print(f"   角色基线: {baseline_info['role_baselines_generated']}")
        print(f"   行业基线: {baseline_info['industry_baselines_generated']}")

        # 数据库上传
        upload_info = summary['database_upload']
        print(f"\n📤 数据库上传:")
        print(f"   全局基线: {'✅' if upload_info['global_uploaded'] else '❌'}")
        print(f"   角色基线: {upload_info['roles_uploaded']} 个")
        print(f"   行业基线: {upload_info['industries_uploaded']} 个")
        print(f"   语义数据: {upload_info.get('total_semantic_uploads', 0)} 个")
        print(f"   向量数据: {upload_info.get('total_vector_uploads', 0)} 个")
        print(f"   错误数量: {upload_info['total_errors']}")

        # 质量指标
        quality_info = summary['quality_metrics']
        print(f"\n📈 质量指标:")
        print(f"   角色覆盖率: {quality_info['classification_coverage']['role_coverage_percent']}%")
        print(f"   行业覆盖率: {quality_info['classification_coverage']['industry_coverage_percent']}%")
        print(f"   平均角色技能: {quality_info['baseline_quality']['avg_skills_per_role']}")
        print(f"   平均行业技能: {quality_info['baseline_quality']['avg_skills_per_industry']}")

        print(f"\n📁 输出目录: {self.config.output_dir}")
        print("=" * 60)
