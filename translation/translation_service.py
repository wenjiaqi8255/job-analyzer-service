#!/usr/bin/env python3
"""
简化版翻译服务 - 直接从Hugging Face下载
适用于GitHub Actions环境
"""

import os
import time
import torch
import hashlib
from transformers import MarianMTModel, MarianTokenizer
from supabase import create_client, Client

class SimpleTranslationService:
    def __init__(self):
        # Supabase连接
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        
        # 模型相关
        self.model = None
        self.tokenizer = None
        self.translation_cache = {}
        
    def load_model(self, model_name="Helsinki-NLP/opus-mt-de-en"):
        """直接从Hugging Face加载模型"""
        print(f"Loading model from Hugging Face: {model_name}")
        print("This may take 3-5 minutes on first run...")
        
        try:
            start_time = time.time()
            
            # 直接下载
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded successfully in {load_time:.1f} seconds")
            
            # 测试模型
            test_result = self.translate_text("Hallo Welt")
            print(f"✅ Model test: 'Hallo Welt' → '{test_result}'")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def detect_language(self, text):
        """简单的语言检测"""
        german_indicators = ['der', 'die', 'das', 'und', 'mit', 'für', 'wir', 'sie', 'ein', 'eine']
        english_indicators = ['the', 'and', 'with', 'for', 'we', 'you', 'a', 'an', 'of', 'to']
        
        text_lower = text.lower()
        german_count = sum(1 for word in german_indicators if word in text_lower)
        english_count = sum(1 for word in english_indicators if word in text_lower)
        
        return 'de' if german_count > english_count else 'en'
    
    def translate_text(self, text, max_length=512):
        """翻译单个文本"""
        if not self.model or not self.tokenizer:
            raise Exception("Model not loaded!")
        
        # 检查缓存
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # 预处理：截断过长文本
            if len(text) > max_length * 4:
                text = text[:max_length * 4]
            
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                                  truncation=True, max_length=max_length)
            
            with torch.no_grad():
                translated = self.model.generate(**inputs, max_length=max_length)
            
            result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # 缓存结果
            self.translation_cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # 翻译失败时返回原文
    
    def fetch_untranslated_jobs(self):
        """获取需要翻译的jobs"""
        try:
            # 查询没有翻译的jobs
            response = self.supabase.table('job_listings')\
                .select('id, description')\
                .is_('translated_description', None)\
                .limit(100)\
                .execute()  # 一次最多处理100个
            
            jobs = response.data
            
            # 过滤出德文jobs
            german_jobs = []
            for job in jobs:
                if self.detect_language(job.get('description', '')) == 'de':
                    german_jobs.append(job)
            
            print(f"Found {len(german_jobs)} German jobs to translate")
            return german_jobs
            
        except Exception as e:
            print(f"Error fetching jobs: {e}")
            return []
    
    def update_translated_job(self, job_id, translated_desc):
        """更新单个job的翻译"""
        try:
            self.supabase.table('job_listings').update({
                'translated_description': translated_desc,
                'translation_completed_at': 'now()'
            }).eq('id', job_id).execute()
            
        except Exception as e:
            print(f"Error updating job {job_id}: {e}")
    
    def run_translation_batch(self):
        """执行完整的批量翻译流程"""
        print("=== Starting Simple Translation Service ===")
        
        # 1. 加载模型
        if not self.load_model():
            print("Failed to load translation model!")
            return False
        
        # 2. 获取待翻译jobs
        german_jobs = self.fetch_untranslated_jobs()
        if not german_jobs:
            print("No jobs to translate!")
            return True
        
        # 3. 逐个翻译并更新
        success_count = 0
        for i, job in enumerate(german_jobs):
            job_id = job['id']
            description = job['description']
            
            print(f"Translating job {i+1}/{len(german_jobs)} (ID: {job_id})")
            
            try:
                # 翻译
                translated = self.translate_text(description)
                
                # 更新数据库
                self.update_translated_job(job_id, translated)
                success_count += 1
                
                print(f"  ✅ Success: {description[:50]}... → {translated[:50]}...")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
        
        print(f"=== Translation completed! {success_count}/{len(german_jobs)} successful ===")
        return True

def main():
    """主函数 - 用于GitHub Actions调用"""
    service = SimpleTranslationService()
    
    # 执行翻译
    success = service.run_translation_batch()
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()