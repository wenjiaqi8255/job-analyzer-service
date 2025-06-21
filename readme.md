python main.py --mode inference --csv sample_data/job_listings_rows.csv

python main.py --mode inference --supabase 

python main.py --run-full-keyword-archiving

GitHub Actions启动
    ↓
1. 数据生命周期管理
   - Day 7: 归档关键词到archive表
   - Day 8: 清理job_listings表
   - Day 180: 清理过期archive数据
    ↓
2. 异常检测分析
   - 获取未分析的新jobs (fetch_jobs_to_analyze)
   - 获取历史数据用于IDF计算 (fetch_idf_corpus)
   - 运行异常检测pipeline
   - 存储结果到数据库
    ↓
3. 缓存更新
   - 检查IDF缓存是否需要更新
   - 必要时重新计算并缓存
    ↓
4. 监控和日志
   - 记录处理统计
   - 性能指标
   - 错误告警


数据流的智能逻辑
IDF计算的数据源选择
开始IDF计算
    ↓
检查archive表数据量
    ↓
if archive_jobs >= 100:
    使用30天历史数据（大样本，稳定）
    重建虚拟文档：关键词 → 文本
else:
    降级到当前数据（小样本，但可用）
    直接使用job descriptions
    ↓
计算TF-IDF权重
存储到缓存
批量处理策略
获取未分析的jobs
    ↓
if jobs_count > 50:
    分批处理（batch_size=50）
    避免内存和时间问题
else:
    一次性处理所有jobs
    ↓
每个job执行异常检测
存储结果到job_anomaly_analysis表
