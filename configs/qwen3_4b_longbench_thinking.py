from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
with read_base():
    from opencompass.configs.datasets.longbenchv2.longbenchv2_gen import \
        LongBenchv2_datasets as LongBenchv2_datasets

datasets=LongBenchv2_datasets


models=[
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen3-4b-turbomind',
        path='/kpfs/intern-legal/model/Qwen3-4B',
        batch_size=8,
        drop_middle=True,
        engine_config=dict(
            gpu_memory_utilization=0.3,
            max_batch_size=8,
            session_len=32768,#32768
            tensor_parallel_size=1,
            tp=1),
        gen_config=dict(
            # max_new_tokens=4096,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            do_sample=True, 
            enable_thinking=True
            ),
        max_out_len=4096, #4096
        max_seq_len=32768,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
        run_cfg=dict(num_gpus=1)
    )
]

work_dir = './results/longbench/qwen3/4B/thinking'