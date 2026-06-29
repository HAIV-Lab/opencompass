"""
==============================================================================
 LiveCodeBench v5 评测 — 自包含版本
==============================================================================
评测集:
  - lcb_code_generation: 代码生成 (pass@1)
  - lcb_code_execution: 代码执行预测 (pass@1)
  - lcb_test_output: 测试输出预测 (pass@1)

时间范围: 2024-08-01 ~ 2025-02-01 (LCB v5)

⚠️ ⚡ 修复说明:
  LiveCodeBench 代码评测使用了嵌套的多进程架构
  (ProcessPoolExecutor → multiprocessing.Process)，在子进程中
  RuntimeModule 可能出现 'NoneType' object has no attribute 'from_string'
  导致所有代码生成结果为 0.00 pass@1。

  修复方法 (已在配置中实现):
  在配置顶部添加了预导入:
    from opencompass.datasets.livecodebench import testing_util
  这样在 fork 子进程时，RuntimeModule 已存在于内存中，
  子进程不需要重新 import，避免了嵌套进程中的 import 异常。

运行:
  python run.py eval_lcbv5_vllm.py
  export COMPASS_DATA_CACHE=/kpfs/dataset/general
==============================================================================
"""

from opencompass.models import VLLMwithChatTemplate
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    LCBCodeGenerationDataset,
    LCBCodeExecutionDataset,
    LCBTestOutputPredictionDataset,
    LCBCodeGenerationEvaluator,
    LCBCodeExecutionEvaluator,
    LCBTestOutputEvaluator,
)

# ⭐ 预导入 testing_util 修复子进程中 RuntimeModule 为 None 的问题
# 在 fork 模式下，子进程继承父进程已导入的模块，避免嵌套子进程中的 import 异常
from opencompass.datasets.livecodebench import testing_util as _lcb_testing_util
assert _lcb_testing_util.RuntimeModule is not None, 'RuntimeModule should not be None'

###########################################################################
#  (1) LCB Code Generation 数据集配置
###########################################################################

CODE_GEN_PROMPT = '### Question:\n{question_content}\n\n{format_prompt}' + \
                  '### Answer: (use the provided format with backticks)\n\n'

lcb_code_generation_datasets = [
    dict(
        type=LCBCodeGenerationDataset,
        abbr='lcb_code_generation',
        # path='opencompass/code_generation_lite',
        path='/kpfs/dataset/general/code_generation_lite',
        release_version='release_v5',
        reader_cfg=dict(
            input_columns=['question_content', 'format_prompt'],
            output_column='question_id',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(role='HUMAN', prompt=CODE_GEN_PROMPT),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(
                type=LCBCodeGenerationEvaluator,
                num_process_evaluate=4,
                timeout=6,
                release_version='release_v5',
                start_date='2024-08-01',
                end_date='2025-02-01',
            ),
            pred_role='BOT',
        ),
        n=1,
    )
]

###########################################################################
#  (2) LCB Code Execution 数据集配置
###########################################################################

lcb_code_execution_datasets = [
    dict(
        type=LCBCodeExecutionDataset,
        abbr='lcb_code_execution',
        # path='opencompass/execution-v2',
        path='/kpfs/dataset/general/execution-v2',
        reader_cfg=dict(
            input_columns=['prompt'],
            output_column='evaluation_sample',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt='You are an expert at Python programming, code execution, test case generation, and fuzzing.',
                        ),
                    ],
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                    ],
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(type=LCBCodeExecutionEvaluator),
            pred_role='BOT',
        ),
        n=1,
    )
]

###########################################################################
#  (3) LCB Test Output 数据集配置
###########################################################################

lcb_test_output_datasets = [
    dict(
        type=LCBTestOutputPredictionDataset,
        abbr='lcb_test_output',
        # path='opencompass/test_generation',
        path='/kpfs/dataset/general/test_generation',
        reader_cfg=dict(
            input_columns=['prompt'],
            output_column='evaluation_sample',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(role='HUMAN', prompt='{prompt}'),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(type=LCBTestOutputEvaluator),
            pred_role='BOT',
        ),
        n=1,
    )
]

# 统一设置最大生成长度
for ds_list in [
    lcb_code_generation_datasets,
    lcb_code_execution_datasets,
    lcb_test_output_datasets,
]:
    for ds in ds_list:
        ds['infer_cfg']['inferencer']['max_out_len'] = 32768

datasets = (
    lcb_code_generation_datasets
    + lcb_code_execution_datasets
    + lcb_test_output_datasets
)

###########################################################################
#  (4) 模型列表
###########################################################################

models = []

# ── Qwen3-4B 非思考模式 (代码生成建议使用 lower temperature) ──
models += [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-4b-nothinking',
        path='/kpfs/intern-legal/model/Qwen3-4B',
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.2),
        max_seq_len=32768,
        max_out_len=32768,
        batch_size=8,
        generation_kwargs=dict(
            temperature=0.0,  # greedy 解码，保证确定性
            do_sample=False,
        ),
        chat_template_kwargs=dict(enable_thinking=False),
        run_cfg=dict(num_gpus=1),
    ),
]

# ── Qwen3-4B 思考模式 ──
models += [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-4b-thinking',
        path='/kpfs/intern-legal/model/Qwen3-4B',
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.2),
        max_seq_len=32768,
        max_out_len=32768,
        batch_size=8,
        generation_kwargs=dict(
            temperature=0.0,
            do_sample=False,
        ),
        chat_template_kwargs=dict(enable_thinking=True),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    ),
]

# ── Qwen2.5-7B-Instruct ──
models += [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2.5-7b-instruct',
        path='/kpfs/intern-legal/model/Qwen2.5-7B-Instruct',
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.3),
        max_seq_len=32768,
        max_out_len=32768,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    ),
]

###########################################################################
#  (5) 推理 & 评测
###########################################################################

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLEvalTask),
    ),
)

###########################################################################
#  (6) 汇总器 & 输出
###########################################################################

summarizer = dict(
    dataset_abbrs=[
        ['lcb_code_generation', 'pass@1'],
        ['lcb_code_execution', 'pass@1'],
        ['lcb_test_output', 'pass@1'],
    ],
)

work_dir = './results/lcbv5_vllm_eval'
