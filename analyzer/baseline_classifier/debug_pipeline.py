#!/usr/bin/env python3
"""
Debug Classification Script
用于调试为什么cybersecurity工作被错误分类为finance的问题
"""

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import sys
import os
from dotenv import load_dotenv
from supabase import create_client

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_baseline_similarity(industry_classifier, embedding_processor, preprocessor):
    """
    测试关键词与baseline的相似度，找出为什么cybersecurity job被分类为finance
    """
    
    print("=" * 80)
    print("BASELINE SIMILARITY DEBUG TEST")
    print("=" * 80)
    
    # 1. 测试风险相关关键词
    risk_related_phrases = [
        "business risks",
        "risk management", 
        "protecting organizations",
        "security incidents",
        "vulnerabilities",
        "cybersecurity",
        "security operations center",
        "blue team",
        "malware analysis",
        "incident response",
        "European leader",
        "performance",
        "customers situation"
    ]
    
    print("\n1. 测试关键词与各baseline的相似度:")
    print("-" * 60)
    
    for phrase in risk_related_phrases:
        try:
            # 预处理短语
            cleaned_phrase = preprocessor.preprocess_text(phrase)
            
            # 获取embedding
            phrase_embedding = embedding_processor.get_embedding(cleaned_phrase, 'debug_test')
            if phrase_embedding is None:
                print(f"❌ {phrase}: 无法获取embedding")
                continue
                
            phrase_embedding_np = phrase_embedding.cpu().numpy().reshape(1, -1)
            
            # 计算与各baseline的相似度
            similarities = {}
            for industry_name, baseline_embedding in industry_classifier.baseline_embeddings.items():
                sim = cosine_similarity(phrase_embedding_np, baseline_embedding)[0][0]
                similarities[industry_name] = sim
            
            # 排序并显示
            sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            sim_str = ", ".join([f"{name}={sim:.4f}" for name, sim in sorted_sims])
            
            winner = sorted_sims[0][0]
            print(f"{'✅' if winner == 'tech' else '❌'} {phrase:25} → {sim_str}")
            
        except Exception as e:
            print(f"❌ {phrase}: Error - {str(e)}")
    
    print("\n" + "=" * 80)
    
    # 2. 测试完整job description的分段相似度
    job_description = """means joining a European leader in Cybersecurity, but above all it means joining specialists who are passionate about protecting organizations that are increasingly exposed, to help them to accomplish their often-essential missions. If Cybersecurity keeps the world go round, then our performance must help to change it for the better. You have a social project you want to support? Job DescriptionAs part of our Blue Team, at the heart of our Security Operations Center (SOC), you will participate in identifying abnormal behavior or vulnerabilities to protect our customers' information systems. Develop and submit a remediation plan adapted to your customers' situation and business risks. Implement requirements and document the tasks performed. Manage alerts & incidents together with an experienced team during night, bank holiday and weekend shifts🤠 Your potential projectsWhat other concrete activities could you take part in if you joined us? Inform the product team of changes to be made to the analysis and incident reporting tools. Guide and advise your customers' operational security team when they deal with a security incident. You are tech-savvy: being able to recognize attacks is vital to stop them. You are hands-on minded and have a basic knowledge of IT security concepts and techniquesYour passionate about Pentesting and IT forensics is a plus (attack analysis, malware behavior, etc.)You are"""
    
    print("\n2. 测试job description分段:")
    print("-" * 60)
    
    # 将description分成几个关键段落
    segments = [
        "European leader in Cybersecurity specialists passionate about protecting organizations",
        "Blue Team Security Operations Center SOC identifying abnormal behavior vulnerabilities",
        "customers situation business risks implement requirements document tasks",
        "alerts incidents experienced team security incident product team analysis",
        "tech-savvy recognize attacks IT security concepts techniques Pentesting forensics malware"
    ]
    
    for i, segment in enumerate(segments, 1):
        try:
            cleaned_segment = preprocessor.preprocess_text(segment)
            segment_embedding = embedding_processor.get_embedding(cleaned_segment, f'debug_segment_{i}')
            
            if segment_embedding is None:
                print(f"❌ Segment {i}: 无法获取embedding")
                continue
                
            segment_embedding_np = segment_embedding.cpu().numpy().reshape(1, -1)
            
            similarities = {}
            for industry_name, baseline_embedding in industry_classifier.baseline_embeddings.items():
                sim = cosine_similarity(segment_embedding_np, baseline_embedding)[0][0]
                similarities[industry_name] = sim
            
            sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            winner = sorted_sims[0][0]
            
            print(f"{'✅' if winner == 'tech' else '❌'} Segment {i} ({winner}): ", end="")
            print(", ".join([f"{name}={sim:.4f}" for name, sim in sorted_sims]))
            print(f"   Text: {segment[:80]}...")
            
        except Exception as e:
            print(f"❌ Segment {i}: Error - {str(e)}")
    
    print("\n" + "=" * 80)
    
    # 3. 检查baseline内容详情
    print("\n3. Baseline详细信息:")
    print("-" * 60)
    
    for industry_name, baseline_embedding in industry_classifier.baseline_embeddings.items():
        print(f"{industry_name}:")
        print(f"  - 向量形状: {baseline_embedding.shape}")
        print(f"  - 向量范围: [{baseline_embedding.min():.4f}, {baseline_embedding.max():.4f}]")
        print(f"  - 向量均值: {baseline_embedding.mean():.4f}")
        print(f"  - 向量标准差: {baseline_embedding.std():.4f}")
    
    print("\n" + "=" * 80)
    
    # 4. 测试去除关键词后的效果
    print("\n4. 测试去除疑似问题词汇的效果:")
    print("-" * 60)
    
    # 原始description
    original_desc = job_description
    
    # 去除business risks相关词汇
    problematic_words = ["business risks", "business", "risks", "customers situation", "performance"]
    cleaned_desc = original_desc
    for word in problematic_words:
        cleaned_desc = cleaned_desc.replace(word, "")
    
    # 测试两个版本
    for desc_name, desc_text in [("原始", original_desc), ("清理后", cleaned_desc)]:
        try:
            processed_desc = preprocessor.preprocess_text(desc_text)
            desc_embedding = embedding_processor.get_embedding(processed_desc, f'debug_{desc_name}')
            
            if desc_embedding is None:
                print(f"❌ {desc_name}: 无法获取embedding")
                continue
                
            desc_embedding_np = desc_embedding.cpu().numpy().reshape(1, -1)
            
            similarities = {}
            for industry_name, baseline_embedding in industry_classifier.baseline_embeddings.items():
                sim = cosine_similarity(desc_embedding_np, baseline_embedding)[0][0]
                similarities[industry_name] = sim
            
            sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            winner = sorted_sims[0][0]
            
            print(f"{'✅' if winner == 'tech' else '❌'} {desc_name}描述 ({winner}): ", end="")
            print(", ".join([f"{name}={sim:.4f}" for name, sim in sorted_sims]))
            
        except Exception as e:
            print(f"❌ {desc_name}: Error - {str(e)}")

    print("\n" + "=" * 80)
    print("测试完成!")


def test_actual_job_classification(pipeline):
    """
    测试实际的job分类流程
    """
    print("\n" + "=" * 80)
    print("ACTUAL JOB CLASSIFICATION TEST")
    print("=" * 80)
    
    # 准备测试数据
    test_job_data = {
        'job_id': '4256087644',
        'job_title': 'WORKING STUDENT SECURITY OPERATIONS CENTER (M/F/D)',
        'company_name': 'Advens',
        'industry': 'Computer and Network Security',
        'effective_description': """means joining a European leader in Cybersecurity, but above all it means joining specialists who are passionate about protecting organizations that are increasingly exposed, to help them to accomplish their often-essential missions. If Cybersecurity keeps the world go round, then our performance must help to change it for the better. You have a social project you want to support? Job DescriptionAs part of our Blue Team, at the heart of our Security Operations Center (SOC), you will participate in identifying abnormal behavior or vulnerabilities to protect our customers' information systems. Develop and submit a remediation plan adapted to your customers' situation and business risks. Implement requirements and document the tasks performed. Manage alerts & incidents together with an experienced team during night, bank holiday and weekend shifts🤠 Your potential projectsWhat other concrete activities could you take part in if you joined us? Inform the product team of changes to be made to the analysis and incident reporting tools. Guide and advise your customers' operational security team when they deal with a security incident. You are tech-savvy: being able to recognize attacks is vital to stop them. You are hands-on minded and have a basic knowledge of IT security concepts and techniquesYour passionate about Pentesting and IT forensics is a plus (attack analysis, malware behavior, etc.)You are"""
    }
    
    print(f"测试Job: {test_job_data['job_title']}")
    print(f"公司: {test_job_data['company_name']}")
    print(f"行业: {test_job_data['industry']}")
    print("-" * 60)
    
    # 运行分类
    try:
        result = pipeline.run(test_job_data)
        print(f"分类结果:")
        print(f"  角色: {result.get('role', 'Unknown')}")
        print(f"  行业: {result.get('industry', 'Unknown')}")
        
        # 如果行业不是tech，则失败
        if result.get('industry') != 'tech':
            print(f"❌ 分类错误! 期望: tech, 实际: {result.get('industry')}")
        else:
            print(f"✅ 分类正确!")
            
    except Exception as e:
        print(f"❌ 分类过程出错: {str(e)}")
        logger.error(f"Classification error: {e}", exc_info=True)


def main():
    from .classification_pipeline import ClassificationPipeline
    from ..config import AppConfig
    from ..utils.resource_manager import ResourceManager
    from ..utils.text_preprocessor import TextPreprocessor
    
    # 加载环境变量
    load_dotenv()
    
    # 获取Supabase配置
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ SUPABASE_URL or SUPABASE_KEY not set in environment.")
        return
    
    try:
        # 创建Supabase客户端
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Successfully connected to Supabase")
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return
    
    # 加载配置
    config = AppConfig()
    
    # 使用ResourceManager加载模型和基线数据
    with ResourceManager(config=config, supabase_client=supabase) as resources:
        embedding_model = resources["embedding_model"]
        nlp_model = resources["nlp_model"]
        semantic_baselines = resources["semantic_baselines"]
        
        if not embedding_model or not nlp_model:
            print("❌ Failed to load necessary models. Exiting pipeline.")
            return
            
        # 初始化文本预处理器
        text_preprocessor = TextPreprocessor(nlp_model)
        
        # 初始化pipeline
        pipeline = ClassificationPipeline(
            embedding_model=embedding_model,
            semantic_baselines=semantic_baselines,
            text_preprocessor=text_preprocessor,
            config=config
        )
        
        # 运行测试
        debug_baseline_similarity(
            pipeline.industry_classifier,
            pipeline.industry_classifier.embedding_processor,
            pipeline.text_preprocessor
        )
        
        test_actual_job_classification(pipeline)

if __name__ == "__main__":
    main()