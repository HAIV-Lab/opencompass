
from opencompass.models import VLLMwithChatTemplate
from mmengine.config import read_base
from opencompass.utils.text_postprocessors import qwen_think_mcq_postprocess

with read_base():
    
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    # from opencompass.configs.models.qwen3.vllm_qwen3_8b_no_thinking import models as Qwen3_8B

# single_subject_abbr = 'ceval-law'
# datasets = [d for d in ceval_datasets if d.get('abbr') == single_subject_abbr]
datasets = ceval_datasets

#####C-Eval数据集下，需要修改qwen_think_mcq_postprocess方法中330行：(?:answer|答案|选项|选择|故选|所以|最终|结论)?   
#####其他数据集，改回(?:answer|答案|选项|选择|故选|所以|最终|结论)
for _dataset in datasets:
    _dataset['eval_cfg']['pred_postprocessor'] = dict(
        type=qwen_think_mcq_postprocess,
        options='ABCD',
    )

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='vllm_qwen3_4b_thinking',
        path='/kpfs/intern-legal/model/Qwen3-4B',
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.3),
        max_seq_len=32768,
        max_out_len=16384,
        batch_size=8,
        generation_kwargs=dict(
            top_p=0.95,temperature=0.6,top_k=20
        ),
        chat_template_kwargs=dict(enable_thinking=True), 
        run_cfg=dict(num_gpus=1),
    )
]

work_dir = './results/ceval/qwen3/4B/thinking'