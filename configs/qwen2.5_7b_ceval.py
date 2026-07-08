from opencompass.models import OpenAISDK
from opencompass.utils.text_postprocessors import qwen_think_mcq_postprocess_ceval
from opencompass.models import VLLMwithChatTemplate
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    # from opencompass.configs.models.qwen2_5 import hf_qwen2_5_72b_instruct

# 选择单科目进行测试（默认为全部科目）
# single_subject_abbr = 'ceval-computer_network'
# datasets = [d for d in ceval_datasets if d.get('abbr') == single_subject_abbr]

datasets = ceval_datasets

if not datasets:
    raise ValueError(
        f'No CEval dataset matched. '
        'Please check available subject names in ceval_gen.'
    )

for _dataset in datasets:
    _dataset['eval_cfg']['pred_postprocessor'] = dict(
        type=qwen_think_mcq_postprocess_ceval,
        options='ABCD',
    )

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2.5-7b-instruct-vllm',
        path='/kpfs/intern-legal/model/Qwen2.5-7B-Instruct',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=4096,
        batch_size=16,
        generation_kwargs=dict(
            temperature=0
            ),
        run_cfg=dict(num_gpus=1),
    )
]

work_dir = './results/ceval/qwen2.5/7B'