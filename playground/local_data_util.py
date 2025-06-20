import pandas as pd

def load_and_process_jobs(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ 成功加载 {len(df)} 条职位数据")
    except Exception as e:
        print(f"❌ 读取CSV文件失败: {e}")
        return pd.DataFrame()
    tech_keywords_high = [
        'software', 'developer', 'programming', 'python', 'java', 'javascript',
        'react', 'vue', 'angular', 'node.js', 'machine learning', 'data science',
        'ai ', 'ml ', 'devops', 'cloud', 'aws', 'azure', 'docker', 'kubernetes'
    ]
    tech_keywords_medium = [
        'technical', 'technology', 'it ', 'computer', 'digital', 'web',
        'mobile', 'app', 'api', 'database', 'sql', 'analytics', 'algorithm'
    ]
    def calculate_tech_score(row):
        title = str(row.get('job_title', '')).lower()
        desc = str(row.get('description', '')).lower()
        combined = title + ' ' + desc
        high_score = sum(2 for keyword in tech_keywords_high if keyword in combined)
        medium_score = sum(1 for keyword in tech_keywords_medium if keyword in combined)
        return high_score + medium_score
    df['tech_score'] = df.apply(calculate_tech_score, axis=1)
    tech_jobs = df[df['tech_score'] >= 2].copy()
    print(f"✅ 筛选出 {len(tech_jobs)} 个技术相关职位")
    return tech_jobs 