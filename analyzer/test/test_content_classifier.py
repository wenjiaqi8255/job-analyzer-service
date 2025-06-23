#!/usr/bin/env python3
"""
轻量级分类器测试代码
测试机器学习方法vs规则方法的效果对比
"""

import re
from typing import List, Tuple
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import time

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightContentClassifier:
    """轻量级内容分类器"""
    
    def __init__(self, model_type='logistic'):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # 限制特征数量保证速度
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,  # 至少出现2次
            max_df=0.95  # 排除过于常见的词
        )
        
        # 选择分类器
        if model_type == 'logistic':
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'naive_bayes':
            self.classifier = MultinomialNB(alpha=0.1)
        elif model_type == 'svm':
            self.classifier = SVC(kernel='linear', probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.model_type = model_type
        self.is_trained = False
        
    def create_training_data(self) -> Tuple[List[str], List[int]]:
        """创建训练数据"""
        
        # 工作要求类内容 (label=1) - 扩展数据集
        job_requirement_examples = [
            # 技术技能
            "Experience with Python programming and web development",
            "Knowledge of SQL databases and data analysis", 
            "Proficiency in JavaScript, React, and Node.js",
            "Understanding of agile development methodologies",
            "Experience with cloud platforms AWS or Azure",
            "Familiarity with version control systems like Git",
            "Knowledge of machine learning algorithms and frameworks",
            "Experience with Docker and container orchestration",
            "Understanding of REST APIs and microservices architecture",
            "Proficiency in Java and Spring framework",
            
            # 教育和经验要求
            "Bachelor's degree in Computer Science or related field",
            "3+ years of experience in software development",
            "5+ years of experience in data science and analytics",
            "Master's degree in Engineering or equivalent experience",
            "Professional certifications in cloud technologies preferred",
            
            # 工作职责
            "Responsible for developing and maintaining applications",
            "You will design and implement scalable software solutions",
            "Lead technical discussions and architecture decisions",
            "Collaborate with cross-functional teams on product development",
            "Conduct code reviews and mentor junior developers",
            "Analyze business requirements and translate to technical solutions",
            "Optimize application performance and database queries",
            "Implement automated testing and CI/CD pipelines",
            
            # 技能要求
            "Strong problem-solving and analytical skills required",
            "Excellent communication and teamwork abilities",
            "Ability to work independently and manage multiple projects",
            "Experience in troubleshooting and debugging complex systems",
            "Knowledge of security best practices and data protection",
            
            # 特定领域技能
            "Experience with Rust programming for systems development",
            "Knowledge of blockchain technologies and smart contracts",
            "Familiarity with DevOps tools and practices",
            "Understanding of data structures and algorithms",
            "Experience with mobile app development iOS or Android",
        ]
        
        # 非工作要求内容 (label=0) - 扩展数据集
        non_requirement_examples = [
            # 公司介绍
            "We are a leading technology company founded in 2010",
            "Our mission is to revolutionize digital experiences",
            "Mindrift is an innovative AI company based in Munich",
            "The company has grown from startup to industry leader",
            "We believe in using collective intelligence ethically",
            "Our values include integrity, excellence, and teamwork",
            "Company culture promotes innovation and collaboration",
            "We are committed to diversity and inclusion",
            "Our headquarters are located in the heart of the city",
            "Founded by former executives from major tech companies",
            
            # 福利待遇
            "We offer competitive salary and comprehensive benefits",
            "Health insurance and dental coverage included",
            "Flexible working hours and remote work options",
            "Professional development opportunities and training budget",
            "Generous vacation policy and paid time off",
            "Stock options and equity participation available",
            "Free meals and snacks in the office",
            "Gym membership and wellness programs",
            "Parental leave and family support benefits",
            "Retirement savings plan with company matching",
            
            # 申请流程
            "To apply, please send your resume to hr@company.com",
            "Apply now through our online portal",
            "Please submit your application with cover letter",
            "Contact our HR team for more information",
            "We look forward to hearing from you",
            "Applications will be reviewed on a rolling basis",
            
            # 营销和宣传语言
            "Join our dynamic team of passionate professionals",
            "Be part of an exciting journey in technology innovation",
            "Exciting opportunity to make a real impact",
            "Work in a fast-paced and cutting-edge environment",
            "Shape the future of artificial intelligence",
            "Join a company that values your growth and development",
            "Become part of our success story",
            
            # 位置和办公环境
            "Located in Munich with excellent public transport",
            "Modern office space with state-of-the-art facilities",
            "Open office environment promoting collaboration",
            "Beautiful campus with outdoor spaces",
            "Multiple office locations across Europe",
        ]
        
        # 组合训练数据
        texts = job_requirement_examples + non_requirement_examples
        labels = [1] * len(job_requirement_examples) + [0] * len(non_requirement_examples)
        
        logger.info(f"Training data: {len(job_requirement_examples)} positive, {len(non_requirement_examples)} negative")
        
        return texts, labels
    
    def train_classifier(self):
        """训练分类器"""
        texts, labels = self.create_training_data()
        
        logger.info(f"Training {self.model_type} classifier...")
        start_time = time.time()
        
        # 向量化
        X = self.vectorizer.fit_transform(texts)
        
        # 训练
        self.classifier.fit(X, labels)
        
        train_time = time.time() - start_time
        self.is_trained = True
        
        logger.info(f"Training completed in {train_time:.3f}s")
        
        # 交叉验证评估
        cv_scores = cross_val_score(self.classifier, X, labels, cv=5, scoring='f1')
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean CV F1 score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return train_time, cv_scores.mean()
    
    def is_job_requirement(self, text: str) -> Tuple[bool, float]:
        """判断文本是否为工作要求"""
        if not self.is_trained:
            raise ValueError("Classifier not trained yet!")
        
        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0][1]  # positive class概率
        is_requirement = proba > 0.6  # 保守阈值
        
        return is_requirement, proba
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[bool, float]]:
        """批量预测"""
        if not self.is_trained:
            raise ValueError("Classifier not trained yet!")
        
        start_time = time.time()
        X = self.vectorizer.transform(texts)
        probas = self.classifier.predict_proba(X)[:, 1]
        predictions = probas > 0.6
        inference_time = time.time() - start_time
        
        logger.info(f"Batch inference ({len(texts)} texts) completed in {inference_time:.3f}s")
        
        return list(zip(predictions, probas))


class RuleBasedFilter:
    """规则过滤器 - 用于对比"""
    
    def __init__(self):
        self.exclusion_patterns = [
            r'(?i)(?:we are|our company is|about us|company overview)',
            r'(?i)(?:founded in|established in|since \d{4})',
            r'(?i)(?:our mission|our vision|our values)',
            r'(?i)(?:we offer|benefits include|compensation)',
            r'(?i)(?:to apply|send your|please submit)',
            r'(?i)(?:located in|based in|headquarters)',
            r'(?i)(?:join our team|be part of|exciting opportunity)',
        ]
    
    def is_job_requirement(self, text: str) -> Tuple[bool, float]:
        """规则判断"""
        if len(text.split()) < 4:
            return False, 0.0
        
        for pattern in self.exclusion_patterns:
            if re.search(pattern, text):
                return False, 0.0
        
        return True, 1.0  # 规则方法没有概率，用1.0表示确定性


def create_test_data():
    """创建测试数据 - 模拟真实job description chunks"""
    
    test_cases = [
        # 明确的工作要求 (应该被保留)
        ("Experience with Python and Django framework", True, "技术技能"),
        ("Bachelor's degree in Computer Science required", True, "教育要求"), 
        ("3+ years of software development experience", True, "经验要求"),
        ("You will be responsible for API development", True, "工作职责"),
        ("Strong knowledge of machine learning algorithms", True, "技术知识"),
        ("Proficiency in React and Node.js preferred", True, "技术技能"),
        ("Understanding of agile development methodologies", True, "方法论知识"),
        
        # 明确的非工作要求 (应该被排除)
        ("We are a leading technology company founded in 2015", False, "公司介绍"),
        ("Our mission is to revolutionize the industry", False, "公司使命"),
        ("We offer competitive salary and benefits", False, "福利待遇"),
        ("To apply, please send your resume", False, "申请流程"),
        ("Join our dynamic team of professionals", False, "营销语言"),
        ("Located in the heart of Munich", False, "位置信息"),
        ("Company culture promotes innovation", False, "公司文化"),
        
        # 边界案例 (可能有分歧)
        ("Strong communication skills required", True, "软技能"),  # 可能被规则误判
        ("Experience working in fast-paced environment", True, "工作环境要求"),  # 边界案例
        ("Opportunity for professional growth", False, "发展机会"),  # 更像福利
        ("Work with cutting-edge technologies", True, "工作内容"),  # 可能是工作要求
        ("Flexible working arrangements available", False, "工作安排"),  # 更像福利
        ("Remote work experience preferred", True, "工作方式要求"),  # 边界案例
    ]
    
    return test_cases


def compare_methods():
    """对比不同方法的效果"""
    
    print("=" * 80)
    print("轻量级分类器 vs 规则方法对比测试")
    print("=" * 80)
    
    # 创建测试数据
    test_cases = create_test_data()
    
    # 初始化方法
    models = {
        'Logistic Regression': LightweightContentClassifier('logistic'),
        'Naive Bayes': LightweightContentClassifier('naive_bayes'),
        'SVM': LightweightContentClassifier('svm'),
        'Rule-based': RuleBasedFilter()
    }
    
    # 训练机器学习模型
    training_results = {}
    for name, model in models.items():
        if hasattr(model, 'train_classifier'):
            train_time, cv_score = model.train_classifier()
            training_results[name] = {'train_time': train_time, 'cv_score': cv_score}
    
    print(f"\n📊 训练结果:")
    for name, results in training_results.items():
        print(f"  {name}: 训练时间 {results['train_time']:.3f}s, CV F1 {results['cv_score']:.3f}")
    
    # 测试所有方法
    print(f"\n📋 测试结果对比 ({len(test_cases)} 个测试案例):")
    print("-" * 100)
    print(f"{'Case':<50} {'Truth':<6} {'LogReg':<8} {'NB':<8} {'SVM':<8} {'Rule':<8} {'Category'}")
    print("-" * 100)
    
    results = {name: {'correct': 0, 'total': 0, 'inference_times': []} for name in models.keys()}
    
    for text, truth, category in test_cases:
        print(f"{text[:48]:<50} {str(truth):<6}", end=" ")
        
        for name, model in models.items():
            start_time = time.time()
            pred, confidence = model.is_job_requirement(text)
            inference_time = time.time() - start_time
            
            results[name]['inference_times'].append(inference_time)
            results[name]['total'] += 1
            if pred == truth:
                results[name]['correct'] += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"{status}({confidence:.2f})<8", end=" ")
        
        print(f"{category}")
    
    # 统计结果
    print("-" * 100)
    print(f"\n📈 性能统计:")
    print(f"{'Method':<20} {'Accuracy':<10} {'Avg Inference Time':<20} {'Speed Rank'}")
    print("-" * 60)
    
    for name, result in results.items():
        accuracy = result['correct'] / result['total']
        avg_time = np.mean(result['inference_times']) * 1000  # 转换为毫秒
        print(f"{name:<20} {accuracy:.1%}        {avg_time:.2f}ms               ", end="")
        
        if avg_time < 0.1:
            print("🚀 极快")
        elif avg_time < 1.0:
            print("⚡ 很快") 
        elif avg_time < 10:
            print("🐎 快")
        else:
            print("🐌 慢")


def test_real_job_descriptions():
    """使用真实job description测试"""
    
    print("\n" + "=" * 80)
    print("真实Job Description测试")
    print("=" * 80)
    
    # 真实的job description
    job_description = """
    About Mindrift:
    We are an innovative AI company founded in 2020, dedicated to revolutionizing artificial intelligence.
    Our mission is to harness collective intelligence for ethical AI development.
    
    Job Title: Senior Software Developer (Rust) - AI Trainer
    
    Key Responsibilities:
    You will be responsible for developing high-performance AI training systems.
    Design and implement scalable machine learning infrastructure.
    Collaborate with research teams to optimize model training pipelines.
    
    Requirements:
    Experience with Rust programming language is essential.
    Strong background in systems programming and performance optimization.
    Knowledge of machine learning frameworks like PyTorch or TensorFlow.
    Bachelor's degree in Computer Science or equivalent experience.
    5+ years of experience in software development.
    
    What We Offer:
    Competitive salary range: €80,000 - €120,000.
    Comprehensive health insurance and dental coverage.
    Flexible working hours and remote work options.
    Professional development budget and conference attendance.
    
    How to Apply:
    Please send your resume and cover letter to careers@mindrift.ai.
    We look forward to hearing from you!
    """
    
    # 简单分块
    chunks = [chunk.strip() for chunk in re.split(r'[.!?]+', job_description) 
              if chunk.strip() and len(chunk.strip()) > 20]
    
    print(f"Job Description 分为 {len(chunks)} 个chunks")
    
    # 初始化最佳模型
    classifier = LightweightContentClassifier('logistic')
    classifier.train_classifier()
    
    rule_filter = RuleBasedFilter()
    
    print(f"\n📋 逐块分析:")
    print("-" * 90)
    print(f"{'Chunk':<60} {'ML':<8} {'Rule':<8} {'Category'}")
    print("-" * 90)
    
    ml_kept = []
    rule_kept = []
    
    for chunk in chunks:
        ml_pred, ml_conf = classifier.is_job_requirement(chunk)
        rule_pred, _ = rule_filter.is_job_requirement(chunk)
        
        # 简单分类chunk类型
        chunk_lower = chunk.lower()
        if any(word in chunk_lower for word in ['mission', 'company', 'founded', 'mindrift']):
            category = "公司介绍"
        elif any(word in chunk_lower for word in ['salary', 'benefits', 'offer', 'insurance']):
            category = "福利待遇"
        elif any(word in chunk_lower for word in ['apply', 'resume', 'send']):
            category = "申请流程"
        elif any(word in chunk_lower for word in ['experience', 'knowledge', 'degree', 'required']):
            category = "技能要求"
        elif any(word in chunk_lower for word in ['responsible', 'design', 'collaborate']):
            category = "工作职责"
        else:
            category = "其他"
        
        if ml_pred:
            ml_kept.append(chunk)
        if rule_pred:
            rule_kept.append(chunk)
        
        ml_status = f"✓({ml_conf:.2f})" if ml_pred else f"✗({ml_conf:.2f})"
        rule_status = "✓" if rule_pred else "✗"
        
        print(f"{chunk[:58]:<60} {ml_status:<8} {rule_status:<8} {category}")
    
    print("-" * 90)
    print(f"\n📊 过滤结果:")
    print(f"  机器学习方法: {len(chunks)} -> {len(ml_kept)} chunks ({len(ml_kept)/len(chunks)*100:.1f}% 保留)")
    print(f"  规则方法:     {len(chunks)} -> {len(rule_kept)} chunks ({len(rule_kept)/len(chunks)*100:.1f}% 保留)")
    
    print(f"\n✅ 机器学习保留的chunks:")
    for i, chunk in enumerate(ml_kept, 1):
        print(f"  {i}. {chunk[:80]}...")
    
    print(f"\n✅ 规则方法保留的chunks:")
    for i, chunk in enumerate(rule_kept, 1):
        print(f"  {i}. {chunk[:80]}...")


if __name__ == "__main__":
    # 运行所有测试
    compare_methods()
    test_real_job_descriptions()
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("🎯 推荐选择:")
    print("  • 如果追求速度: 使用规则方法")
    print("  • 如果追求准确率: 使用Logistic Regression")
    print("  • 如果内存受限: 使用Naive Bayes")
    print("  • 如果要最佳效果: 规则 + 机器学习组合")
    
    print("\n💡 集成建议:")
    print("  1. 先用规则快速过滤明显案例")
    print("  2. 对边界案例使用机器学习分类")
    print("  3. 这样可以兼顾速度和准确率")
    
    print("\n🎉 测试完成!")