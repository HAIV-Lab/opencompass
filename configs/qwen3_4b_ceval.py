
from opencompass.utils.text_postprocessors import qwen_think_mcq_postprocess_ceval
from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate

with read_base():
    
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    # from opencompass.configs.models.qwen3.vllm_qwen3_8b_no_thinking import models as Qwen3_8B

# single_subject_abbr = 'ceval-probability_and_statistics'
# datasets = [d for d in ceval_datasets if d.get('abbr') == single_subject_abbr]
datasets = ceval_datasets

for _dataset in datasets:
    _dataset['eval_cfg']['pred_postprocessor'] = dict(
        type=qwen_think_mcq_postprocess_ceval,
        options='ABCD',
    )

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='vllm_qwen3_4b_no_thinking',
        path='/kpfs/intern-legal/model/Qwen3-4B',
        max_seq_len=32768,
        max_out_len=16384,
        batch_size=16,
        generation_kwargs=dict(
            temperature=0.0
        ),
        chat_template_kwargs=dict(enable_thinking=False),  # <-- 在这里配置 thinking
        run_cfg=dict(num_gpus=1),
    )
]

work_dir = './results/ceval/qwen3/4B/non_thinking'