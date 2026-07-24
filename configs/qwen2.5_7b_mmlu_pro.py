from opencompass.models import VLLMwithChatTemplate
from mmengine.config import read_base
from opencompass.utils.text_postprocessors import qwen_think_mcq_postprocess

with read_base():
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets

datasets=mmlu_pro_datasets

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='vllm_qwen2.5_7b',
        path='/kpfs/user/wtt/sft_data_analyise/models/infinity_7m_core_clean',
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.5,enable_lora=True),
        max_seq_len=32768,
        max_out_len=2048,
        batch_size=16,
        generation_kwargs=dict(
            temperature=0.0
        ),
        run_cfg=dict(num_gpus=1),
    )
]

work_dir = './results/mmlu_pro/qwen2.5/7B/infinity_clean/'





