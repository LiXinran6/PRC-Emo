import json
import numpy as np
import random
from datasets import Dataset
from typing import List, Dict, Any
import argparse

class DialogSample:
    """对话样本类，用于计算对话难度"""
    
    def __init__(self, sample_data: Dict[str, Any], dataset_name: str = 'iemocap'):
        self.sample_data = sample_data
        self.dataset_name = dataset_name.upper()
        self.messages = sample_data.get('messages', [])
        
        # 难度计算相关属性
        self.numberofutterances = 0
        self.numberofemotionshifts = 0
        self.numberofspeakers = 0
        self.emotion_shift_weighted = 0
        self.difficulty = 0
        
        # 情感标签映射
        self.emotion_maps = {
            'IEMOCAP': {
                'excitement': 'excitement', 'excited': 'excitement',
                'neutral': 'neutral',
                'frustration': 'frustration', 'frustrated': 'frustration',
                'sadness': 'sadness', 'sad': 'sadness',
                'happiness': 'happiness', 'happy': 'happiness', 'joy': 'happiness',
                'anger': 'anger', 'angry': 'anger'
            },
            'MELD': {
                'neutral': 'neutral',
                'surprise': 'surprise', 'surprised': 'surprise',
                'fear': 'fear', 'scared': 'fear',
                'sadness': 'sadness', 'sad': 'sadness',
                'joy': 'joy', 'happy': 'joy', 'happiness': 'joy',
                'disgust': 'disgust', 'disgusted': 'disgust',
                'anger': 'anger', 'angry': 'anger'
            }
        }
        
        # 计算难度
        self.calculate_difficulty()
    
    def extract_emotion_sequences(self) -> Dict[str, List[str]]:
        """从messages中提取每个说话者的情感序列"""
        speaker_emotions = {}
        
        for msg in self.messages:
            if msg.get('role') == 'assistant':
                # 从助手回复中提取情感标签
                emotion = msg.get('content', '').strip().lower()
                # 标准化情感标签
                emotion_map = self.emotion_maps.get(self.dataset_name, {})
                emotion = emotion_map.get(emotion, emotion)
                
                # 假设每个对话样本对应一个说话者（可根据实际情况调整）
                speaker_id = 'speaker_1'  # 可以根据实际需求修改
                
                if speaker_id not in speaker_emotions:
                    speaker_emotions[speaker_id] = []
                speaker_emotions[speaker_id].append(emotion)
        
        return speaker_emotions
    
    def get_similarity_matrix(self):
        """获取情感相似度矩阵"""
        # 情感坐标映射（基于Russell's Circumplex Model）
        emotion_positions = {
            "excitement": (np.cos(7 * np.pi / 20), np.sin(7 * np.pi / 20)),
            "happiness": (np.cos(3 * np.pi / 20), np.sin(3 * np.pi / 20)),
            "joy": (np.cos(3 * np.pi / 20), np.sin(3 * np.pi / 20)),
            "anger": (-np.cos(9 * np.pi / 20), np.sin(9 * np.pi / 20)),
            "frustration": (-np.cos(np.pi / 20), -np.sin(np.pi / 20)),
            "sadness": (-np.cos(9 * np.pi / 20), -np.sin(9 * np.pi / 20)),
            "fear": (-np.cos(np.pi / 20), np.sin(np.pi / 20)),
            "disgust": (-np.cos(3 * np.pi / 20), np.sin(3 * np.pi / 20)),
            "surprise": (np.cos(9 * np.pi / 20), np.sin(9 * np.pi / 20)),
            "neutral": (0, 0)
        }
        
        emotions = list(emotion_positions.keys())
        n = len(emotions)
        similarity_matrix = np.zeros((n, n))
        emotion_to_index = {emotion: idx for idx, emotion in enumerate(emotions)}
        
        def cosine_similarity(p1, p2):
            dot_product = np.dot(p1, p2)
            norm_p1 = np.linalg.norm(p1)
            norm_p2 = np.linalg.norm(p2)
            if norm_p1 == 0 or norm_p2 == 0:
                return 0.0
            return dot_product / (norm_p1 * norm_p2)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    p1 = emotion_positions[emotions[i]]
                    p2 = emotion_positions[emotions[j]]
                    v1, v2 = p1[0], p2[0]  # valence values
                    
                    if v1 * v2 < 0:  # 相反极性
                        similarity_matrix[i][j] = 0
                    elif v1 * v2 == 0:  # 中性
                        similarity_matrix[i][j] = 1 / n
                    else:
                        similarity_matrix[i][j] = max(cosine_similarity(np.array(p1), np.array(p2)), 0)
        
        return similarity_matrix, emotion_to_index
    
    def calculate_difficulty(self):
        """计算对话难度"""
        speaker_emotions = self.extract_emotion_sequences()
        
        # 计算基本统计信息
        self.numberofutterances = len([msg for msg in self.messages if msg.get('role') == 'user'])
        self.numberofspeakers = len(speaker_emotions)
        
        # 获取相似度矩阵
        similarity_matrix, emotion_to_index = self.get_similarity_matrix()
        
        # 计算情感转换难度
        k = 1  # 相似度权重
        b = 0.4  # 基础偏置
        
        for speaker_id, emotions in speaker_emotions.items():
            for i in range(len(emotions) - 1):
                current_emo = emotions[i]
                next_emo = emotions[i + 1]
                
                if (current_emo != next_emo and 
                    current_emo in emotion_to_index and 
                    next_emo in emotion_to_index):
                    
                    self.numberofemotionshifts += 1
                    current_idx = emotion_to_index[current_emo]
                    next_idx = emotion_to_index[next_emo]
                    
                    similarity_score = abs(similarity_matrix[current_idx][next_idx]) * k + b
                    self.emotion_shift_weighted += similarity_score
        
        # 计算最终难度分数
        denominator = self.numberofutterances + self.numberofspeakers
        if denominator > 0:
            self.difficulty = (self.emotion_shift_weighted + self.numberofspeakers) / denominator
        else:
            self.difficulty = 0


class CurriculumLearningManager:
    """课程学习管理器"""
    
    def __init__(self, args):
        self.args = args
        self.bucket_number = getattr(args, 'bucket_number', 8)
        self.dataset_name = getattr(args, 'data_name', 'iemocap')
        
    def prepare_curriculum_data(self, dataset, current_step: int = 1):
        """
        准备课程学习数据
        
        Args:
            dataset: 原始数据集
            current_step: 当前训练步骤(1到bucket_number)
        
        Returns:
            处理后的数据集
        """
        # 将HuggingFace Dataset转换为样本列表
        samples = []
        for item in dataset:
            dialog_sample = DialogSample(item, self.dataset_name)
            samples.append({
                'sample': item,
                'difficulty': dialog_sample.difficulty,
                'dialog_obj': dialog_sample
            })
        
        # 按难度排序
        samples.sort(key=lambda x: x['difficulty'])
        
        print(f"Total samples: {len(samples)}")
        print(f"Difficulty range: {samples[0]['difficulty']:.4f} - {samples[-1]['difficulty']:.4f}")
        
        # 创建桶
        buckets = self.create_buckets(samples)
        
        # 根据当前步骤选择数据
        selected_samples = self.select_samples_by_step(buckets, current_step)
        
        print(f"Step {current_step}/{self.bucket_number}: Using {len(selected_samples)} samples")
        
        # 转换回HuggingFace Dataset格式
        selected_data = [item['sample'] for item in selected_samples]
        return Dataset.from_list(selected_data)
    
    def create_buckets(self, samples: List[Dict]) -> List[List[Dict]]:
        """将样本分成桶"""
        bucket_size = (len(samples) + self.bucket_number - 1) // self.bucket_number
        buckets = []
        
        for i in range(0, len(samples), bucket_size):
            bucket = samples[i:i + bucket_size]
            buckets.append(bucket)
        
        print(f"Created {len(buckets)} buckets with sizes: {[len(b) for b in buckets]}")
        return buckets
    
    def select_samples_by_step(self, buckets: List[List[Dict]], step: int) -> List[Dict]:
        """根据训练步骤选择样本"""
        selected_samples = []
        
        # 逐步增加难度：step=1只用第一个桶，step=2用前两个桶，以此类推
        for i in range(min(step, len(buckets))):
            selected_samples.extend(buckets[i])
        
        # 打乱选中的样本
        random.shuffle(selected_samples)
        return selected_samples
    
    def get_curriculum_schedule(self, total_steps: int) -> List[int]:
        """
        获取课程学习的调度方案
        
        Args:
            total_steps: 总训练步数
            
        Returns:
            每个训练步对应的课程步骤
        """
        steps_per_curriculum = total_steps // self.bucket_number
        schedule = []
        
        for curriculum_step in range(1, self.bucket_number + 1):
            start_step = (curriculum_step - 1) * steps_per_curriculum
            end_step = curriculum_step * steps_per_curriculum
            
            if curriculum_step == self.bucket_number:
                end_step = total_steps  # 最后一个阶段包含所有剩余步骤
            
            for step in range(start_step, end_step):
                schedule.append(curriculum_step)
        
        return schedule


def create_curriculum_datasets(train_dataset, args, curriculum_manager=None):
    """
    创建所有课程学习阶段的数据集
    
    Args:
        train_dataset: 原始训练数据集
        args: 参数配置
        curriculum_manager: 课程学习管理器
        
    Returns:
        Dict[int, Dataset]: 每个课程步骤对应的数据集
    """
    if curriculum_manager is None:
        curriculum_manager = CurriculumLearningManager(args)
    
    curriculum_datasets = {}
    
    for step in range(1, curriculum_manager.bucket_number + 1):
        curriculum_datasets[step] = curriculum_manager.prepare_curriculum_data(
            train_dataset, current_step=step
        )
    
    return curriculum_datasets


# 使用示例
def example_usage():
    """使用示例"""
    # 模拟参数
    args = argparse.Namespace(
        bucket_number=4,
        data_name='iemocap',
        curriculum=True
    )
    
    # 模拟数据
    sample_data = [
        {
            'messages': [
                {'role': 'system', 'content': 'You are an emotion classifier.'},
                {'role': 'user', 'content': 'I am so happy today!'},
                {'role': 'assistant', 'content': 'happiness'}
            ]
        },
        {
            'messages': [
                {'role': 'system', 'content': 'You are an emotion classifier.'},
                {'role': 'user', 'content': 'This makes me angry and frustrated!'},
                {'role': 'assistant', 'content': 'anger'}
            ]
        }
    ]
    
    # 创建数据集
    from datasets import Dataset
    dataset = Dataset.from_list(sample_data)
    
    # 创建课程学习管理器
    curriculum_manager = CurriculumLearningManager(args)
    
    # 创建课程数据集
    curriculum_datasets = create_curriculum_datasets(dataset, args, curriculum_manager)
    
    # 获取调度方案
    schedule = curriculum_manager.get_curriculum_schedule(total_steps=1000)
    
    print(f"Curriculum datasets created: {list(curriculum_datasets.keys())}")
    print(f"Schedule for first 20 steps: {schedule[:20]}")


if __name__ == "__main__":
    example_usage()