import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from os import environ
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

@LOAD_DATASET.register_module()
class MMLUReduxDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path)
        dataset = DatasetDict()
        if environ.get('DATASET_SOURCE') == 'HF':
            print("=====load from hf=========")
            try:
                dataset_dict = DatasetDict()
                for split in ['test']:
                    print(f"path:{path},name:{name},split:{split}")
                    ms_dataset = load_dataset(path, name=name, split=split)
                    print("-----------------------load finish-------------")
                    dataset_list = []
                    for line in ms_dataset:
                        print("---------------convert data-----------------")
                        sample = MMLUReduxDataset._convert_to_opencompass(line)
                        print(f"=============sample:{sample}==================")
                        dataset_list.append(sample)
                    dataset_dict[split] = Dataset.from_list(dataset_list)
                return dataset_dict
            except Exception as e:
                print(f"hfmirror 加载失败: {e}, 请检查...")
            

    @staticmethod
    def _convert_to_opencompass(record: dict) -> dict:
        """将 EvalScope 格式转换为 OpenCompass 格式"""
        error_type = record.get('error_type', '')
        choices = record.get('choices', [''] * 4)
        target_indices = [int(record.get('answer', 0))]
        correct_answer = record.get('correct_answer', None)
        
        # 根据 error_type 处理校正逻辑
        if error_type == 'no_correct_answer' and correct_answer:
            # 没有正确选项：替换被标记为错误的选项
            if 0 <= target_indices[0] < len(choices):
                choices[target_indices[0]] = correct_answer
                
        elif error_type == 'wrong_groundtruth' and correct_answer:
            # 错误的标准答案：使用校正后的答案
            try:
                target_indices = [int(correct_answer)]
            except ValueError:
                choice_index = ord(correct_answer.upper()) - ord('A')
                if 0 <= choice_index < len(choices):
                    target_indices = [choice_index]
                    
        elif error_type == 'multiple_correct_answers' and correct_answer:
            # 多个正确选项：处理逗号分隔的答案
            correct_answer = correct_answer.strip('()')
            correct_answer = correct_answer.replace(' and ', ',').replace(' or ', ',')
            try:
                target_indices = list(map(int, correct_answer.split(',')))
            except ValueError:
                try:
                    target_indices = [ord(c.upper()) - ord('A') 
                                     for c in correct_answer.split(',')]
                except (ValueError, TypeError):
                    # 在 choices 中查找匹配的选项
                    target_indices = []
                    for ans in correct_answer.split(','):
                        ans = ans.strip()
                        if ans in choices:
                            target_indices.append(choices.index(ans))
        
        # 构建 OpenCompass 标准格式：将选项展开为独立字段 A, B, C, D
        if target_indices:
            # 遍历 target_indices，将每个索引转换为对应的大写字母
            target_letters = ['ABCD'[idx] if 0 <= idx < 4 else '?' for idx in target_indices]
        else:
            target_letters = ['A']  # 默认值
        result = {
            'input': record.get('question', ''),
            'target': target_letters,
        }
        
        # 展开选项
        option_letters = ['A', 'B', 'C', 'D']
        for i, choice in enumerate(choices):
            if i < len(option_letters):
                result[option_letters[i]] = choice
        
        # 可选：保存元数据供调试使用
        result['_metadata'] = {
            'error_type': error_type,
            'correct_answer': correct_answer,
            'potential_reason': record.get('potential_reason', '')
        }
        
        return result