from typing import Dict, List

class SkillProcessor:
    """技能预处理和标准化"""
    
    def __init__(self):
        self.skill_types = {
            'programming_language': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'scala', 'kotlin', 'swift'],
            'framework': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'laravel', 'express'],
            'database': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra'],
            'cloud_service': ['aws', 'azure', 'google_cloud', 'docker', 'kubernetes'],
            'tool': ['figma', 'sketch', 'photoshop', 'tableau', 'jira', 'confluence'],
            'methodology': ['agile', 'scrum', 'lean', 'kanban', 'devops'],
            'certification': ['certified', 'certification', 'associate', 'professional'],
            'language': ['english', 'german', 'french', 'spanish', 'chinese', 'japanese']
        }
    
    def extract_all_skills(self, skills_vocab: Dict) -> List[Dict]:
        """从技能词汇中提取所有技能"""
        all_skills = []
        
        def extract_recursive(data, path="", context=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    new_context = f"{context} {key}".strip()
                    if isinstance(value, list):
                        for skill in value:
                            all_skills.append({
                                'skill': skill,
                                'path': new_path,
                                'context': new_context,
                                'skill_type': self._infer_skill_type(skill, key)
                            })
                    else:
                        extract_recursive(value, new_path, new_context)
        
        extract_recursive(skills_vocab)
        return all_skills
    
    def _infer_skill_type(self, skill: str, category: str) -> str:
        """推断技能类型"""
        skill_lower = skill.lower()
        category_lower = category.lower()
        
        for skill_type, keywords in self.skill_types.items():
            if any(keyword in skill_lower for keyword in keywords) or skill_type in category_lower:
                return skill_type
        
        return 'domain_knowledge' 