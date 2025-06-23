# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# import pickle

# class ContentClassifier:
#     def __init__(self):
#         self.vectorizer = TfidfVectorizer(
#             max_features=1000,  # 限制特征数量，保证速度
#             ngram_range=(1, 2),
#             stop_words='english'
#         )
#         # 选择最快的模型之一
#         self.classifier = LogisticRegression(random_state=42)
        
#     def create_training_data(self):
#         """快速创建训练数据 - 不需要大量标注"""
        
#         # 工作要求类内容 (label=1)
#         job_requirement_examples = [
#             "Experience with Python programming and web development",
#             "Knowledge of SQL databases and data analysis", 
#             "Bachelor's degree in Computer Science or related field",
#             "3+ years of experience in software development",
#             "Proficiency in JavaScript, React, and Node.js",
#             "Understanding of agile development methodologies",
#             "Strong problem-solving and analytical skills",
#             "Experience with cloud platforms AWS or Azure",
#             "Familiarity with version control systems like Git",
#             "Responsible for developing and maintaining applications"
#         ]
        
#         # 非工作要求内容 (label=0)  
#         non_requirement_examples = [
#             "We are a leading technology company founded in 2010",
#             "Our mission is to revolutionize digital experiences",
#             "Join our dynamic team of passionate professionals",
#             "We offer competitive salary and comprehensive benefits",
#             "Our office is located in the heart of Munich",
#             "Company culture promotes innovation and collaboration", 
#             "To apply, please send your resume to hr@company.com",
#             "We provide flexible working hours and remote options",
#             "Our values include integrity, excellence, and teamwork",
#             "The company has grown from startup to industry leader"
#         ]
        
#         # 组合训练数据
#         texts = job_requirement_examples + non_requirement_examples
#         labels = [1] * len(job_requirement_examples) + [0] * len(non_requirement_examples)
        
#         return texts, labels
    
#     def train_quick_classifier(self):
#         """快速训练分类器"""
#         texts, labels = self.create_training_data()
        
#         # 训练
#         X = self.vectorizer.fit_transform(texts)
#         self.classifier.fit(X, labels)
        
#         print(f"Trained classifier with {len(texts)} examples")
        
#     def is_job_requirement(self, text: str) -> tuple[bool, float]:
#         """判断文本是否为工作要求"""
#         X = self.vectorizer.transform([text])
#         proba = self.classifier.predict_proba(X)[0][1]  # 获取positive class概率
#         is_requirement = proba > 0.6  # 保守阈值
        
#         return is_requirement, proba
    
#     def save_model(self, path: str):
#         """保存模型"""
#         with open(path, 'wb') as f:
#             pickle.dump({
#                 'vectorizer': self.vectorizer,
#                 'classifier': self.classifier
#             }, f)
    
#     def load_model(self, path: str):
#         """加载模型"""
#         with open(path, 'rb') as f:
#             models = pickle.load(f)
#             self.vectorizer = models['vectorizer']
#             self.classifier = models['classifier']