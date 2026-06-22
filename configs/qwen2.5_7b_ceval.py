from opencompass.models import OpenAISDK
from opencompass.utils.text_postprocessors import qwen_think_mcq_postprocess

from mmengine.config import read_base

with read_base():
    
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    from opencompass.configs.models.qwen2_5 import hf_qwen2_5_72b_instruct

# 选择单科目进行测试（默认为全部科目）
# single_subject_abbr = 'ceval-computer_network'
# datasets = [d for d in ceval_datasets if d.get('abbr') == single_subject_abbr]

datasets = ceval_datasets

if not datasets:
    raise ValueError(
        f'No CEval dataset matched abbr={single_subject_abbr}. '
        'Please check available subject names in ceval_gen.'
    )

for _dataset in datasets:
    _dataset['eval_cfg']['pred_postprocessor'] = dict(
        type=qwen_think_mcq_postprocess,
        options='ABCD',
    )

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN', begin='<用户>:', end='<\用户>\n'),
        dict(role='BOT', api_role='BOT', begin='<助手>:', end='<\助手>\n', generate=True),
    ],
    reserved_roles=[
        dict(
            role='SYSTEM',
            api_role='SYSTEM',
            prompt=(
                "你是一个选择题答题助手。\n"
                "你必须严格遵守以下规则：\n"
                "1. 只能输出一个大写字母（A/B/C/D）\n"
                "2. 禁止输出任何解释、推理或额外内容\n"
                "3. 禁止输出任何标点、单词或句子\n"
                "4. 输出必须是 A 或 B 或 C 或 D 中的一个\n"
            ),
            begin='<系统>: ', 
            end='<\系统>\n',
            ),
    ],
)


models = [hf_qwen2_5_72b_instruct]

work_dir = './results/ceval/qwen2.5'