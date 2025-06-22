import json
import hashlib
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os
from supabase import create_client, Client

# 加载环境变量
load_dotenv()

# Supabase配置
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("请确保设置了SUPABASE_URL和SUPABASE_KEY环境变量")

# 初始化Supabase客户端
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def calculate_hash(data: dict) -> str:
    """计算数据的SHA-256哈希值"""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def import_global_baseline():
    """导入全局基线数据"""
    print("\n正在处理全局基线数据...")
    # 读取JSON文件
    json_path = Path("/Users/wenjiaqi/Downloads/job_analyzer_service/sample_data/final_data/output/global_baseline.json")
    with open(json_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    
    # 计算数据哈希值
    source_hash = calculate_hash(baseline_data)
    
    # 准备插入数据
    insert_data = {
        "baseline_type": "global",
        "name": "global_baseline",
        "version": 1,
        "is_active": True,
        "baseline_data": baseline_data,
        "source_hash": source_hash
    }
    
    try:
        # 检查是否已存在相同的数据
        result = supabase.table("semantic_baselines") \
            .select("*") \
            .eq("source_hash", source_hash) \
            .execute()
        
        if result.data:
            print("该baseline数据已存在，跳过导入")
            return
        
        # 插入新数据
        result = supabase.table("semantic_baselines") \
            .insert(insert_data) \
            .execute()
        
        print("成功导入global baseline数据")
        print(f"插入的记录ID: {result.data[0]['id']}")
        
    except Exception as e:
        print(f"导入数据时发生错误: {str(e)}")

def import_industry_baselines():
    """导入行业基线数据"""
    print("\n正在处理行业基线数据...")
    # 读取JSON文件
    json_path = Path("/Users/wenjiaqi/Downloads/job_analyzer_service/sample_data/final_data/output/industry_baselines.json")
    with open(json_path, "r", encoding="utf-8") as f:
        all_industry_data = json.load(f)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 遍历每个行业并分别导入
    for industry, baseline_data in all_industry_data.items():
        # 计算数据哈希值
        source_hash = calculate_hash(baseline_data)
        
        # 准备插入数据
        insert_data = {
            "baseline_type": "industry",
            "name": f"{industry}_baseline",
            "version": 1,
            "is_active": True,
            "baseline_data": baseline_data,
            "source_hash": source_hash
        }
        
        try:
            # 检查是否已存在相同的数据
            result = supabase.table("semantic_baselines") \
                .select("*") \
                .eq("source_hash", source_hash) \
                .execute()
            
            if result.data:
                print(f"行业 {industry} 的baseline数据已存在，跳过导入")
                skip_count += 1
                continue
            
            # 插入新数据
            result = supabase.table("semantic_baselines") \
                .insert(insert_data) \
                .execute()
            
            print(f"成功导入 {industry} 行业的baseline数据")
            print(f"插入的记录ID: {result.data[0]['id']}")
            success_count += 1
            
        except Exception as e:
            print(f"导入 {industry} 行业数据时发生错误: {str(e)}")
            error_count += 1
    
    print(f"\n行业基线导入统计:")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"错误: {error_count}")

def import_role_baselines():
    """导入角色基线数据"""
    print("\n正在处理角色基线数据...")
    # 读取JSON文件
    json_path = Path("/Users/wenjiaqi/Downloads/job_analyzer_service/sample_data/final_data/output/role_baselines.json")
    with open(json_path, "r", encoding="utf-8") as f:
        all_role_data = json.load(f)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 遍历每个角色并分别导入
    for role, baseline_data in all_role_data.items():
        # 计算数据哈希值
        source_hash = calculate_hash(baseline_data)
        
        # 准备插入数据
        insert_data = {
            "baseline_type": "role",
            "name": f"{role}_baseline",
            "version": 1,
            "is_active": True,
            "baseline_data": baseline_data,
            "source_hash": source_hash
        }
        
        try:
            # 检查是否已存在相同的数据
            result = supabase.table("semantic_baselines") \
                .select("*") \
                .eq("source_hash", source_hash) \
                .execute()
            
            if result.data:
                print(f"角色 {role} 的baseline数据已存在，跳过导入")
                skip_count += 1
                continue
            
            # 插入新数据
            result = supabase.table("semantic_baselines") \
                .insert(insert_data) \
                .execute()
            
            print(f"成功导入 {role} 角色的baseline数据")
            print(f"插入的记录ID: {result.data[0]['id']}")
            success_count += 1
            
        except Exception as e:
            print(f"导入 {role} 角色数据时发生错误: {str(e)}")
            error_count += 1
    
    print(f"\n角色基线导入统计:")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"错误: {error_count}")

if __name__ == "__main__":
    print("=== 开始导入所有基线数据 ===")
    
    # 导入全局基线
    print("\n1. 导入Global Baseline数据...")
    import_global_baseline()
    
    # 导入行业基线
    print("\n2. 导入Industry Baselines数据...")
    import_industry_baselines()
    
    # 导入角色基线
    print("\n3. 导入Role Baselines数据...")
    import_role_baselines()
    
    print("\n=== 所有基线数据导入完成 ===") 