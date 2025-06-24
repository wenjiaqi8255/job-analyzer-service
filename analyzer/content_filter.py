import re
from typing import List

class ContentFilter:
    def __init__(self):
        self.exclusion_patterns = [
            # 公司介绍 - 注释掉过于宽泛的规则
            # r'(?i)(?:we are|our company is|about us|company overview|we)',
            # r'(?i)(?:founded in|established in|since \d{4})',
            # r'(?i)(?:our mission|our vision|our values|our values|company culture|our)',
            
            # 福利待遇 - 保留明确的福利待遇讨论，但移除通用的
            r'(?i)(?:we offer|benefits include|compensation package)',
            r'(?i)(?:salary range|competitive salary|attractive salary)',
            # r'(?i)(?:health insurance|dental coverage|pension plan)',
            
            # 申请流程
            r'(?i)(?:to apply|send your|please submit|apply now)',
            r'(?i)(?:contact us|reach out|get in touch)',
            
            # 位置信息 - 过于宽泛，可能出现在职位描述中
            # r'(?i)(?:located in|based in|headquarters|office)',
            # r'(?i)(?:address|street|city|region|country)',
            
            # 营销语言 - 注释掉，因为它们可能与职位描述混合在一起
            # r'(?i)(?:join our team|be part of|exciting opportunity)',
            # r'(?i)(?:dynamic environment|fast.paced|cutting.edge)',
        ]
    
    def filter_job_requirement_chunks(self, chunks: list[str]) -> list[str]:
        """过滤出工作要求相关的chunks"""
        filtered_chunks = []
        
        for chunk in chunks:
            if self._should_keep_chunk(chunk):
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def _should_keep_chunk(self, chunk: str) -> bool:
        """判断是否应该保留此chunk"""
        # # 过短的chunk直接排除
        # if len(chunk.split()) < 4:
        #     return False
        
        # 检查排除模式
        for pattern in self.exclusion_patterns:
            if re.search(pattern, chunk):
                return False
        
        return True
