from opencompass.models import VLLMwithChatTemplate
from mmengine.config import read_base
from opencompass.utils.text_postprocessors import qwen_think_mcq_postprocess

with read_base():
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets

datasets=mmlu_pro_datasets

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='vllm_qwen3_4b_thinking',
        path='/kpfs/intern-legal/model/Qwen3-4B',
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.3),
        max_seq_len=32768,
        max_out_len=16384,
        batch_size=16,
        generation_kwargs=dict(
            top_k=20,temperature=0.6,top_p=0.95
        ),
        chat_template_kwargs=dict(enable_thinking=True),  # <-- 在这里配置 thinking
        run_cfg=dict(num_gpus=1),
    )
]

work_dir = './results/mmlu_pro/qwen3/4B/thinking'