
from opencompass.models import VLLMwithChatTemplate
from mmengine.config import read_base
from opencompass.utils.text_postprocessors import qwen_think_mcq_postprocess

with read_base():
    
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    # from opencompass.configs.models.qwen3.vllm_qwen3_8b_no_thinking import models as Qwen3_8B

# single_subject_abbr = 'lukaemon_mmlu_college_chemistry'
# datasets = [d for d in mmlu_datasets if d.get('abbr') == single_subject_abbr]
datasets = mmlu_datasets

for _dataset in datasets:
    _dataset['eval_cfg']['pred_postprocessor'] = dict(
        type=qwen_think_mcq_postprocess,
        options='ABCD',
    )

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='vllm_qwen2.5_7b',
        path='/kpfs/intern-legal/model/Qwen2.5-7B-Instruct',
        # model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.5),
        max_seq_len=32768,
        max_out_len=2048,
        batch_size=16,
        generation_kwargs=dict(
            temperature=0.0
        ),
        # chat_template_kwargs=dict(enable_thinking=False),  # <-- 在这里配置 thinking
        run_cfg=dict(num_gpus=1),
    )
]

work_dir = './results/mmlu/qwen2.5/7B'