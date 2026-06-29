"""
==============================================================================
 Instruction Following 评测 (IFEval + IFBench + Multi-IF) — 自包含版本
==============================================================================
评测集:
  - IFEval: 通用指令跟随 (Prompt-level / Inst-level, strict / loose)
  - IFBench: 更强的指令跟随评测 (含关键词计数、格式约束等)
  - Multi-IF: 多语言多轮指令跟随 (仅中英文子集，多轮推理)

运行:
  python run.py eval_instruction_following_vllm.py
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
from opencompass.datasets import (IFEvalDataset, IFEvaluator, IFBenchEvaluator,
                                   MultiIFDataset, MultiIFInferencer,
                                   MultiIFEvaluator)

###########################################################################
#  (1) IFEval 数据集配置
###########################################################################

ifeval_datasets = [
    dict(
        abbr='IFEval',
        type=IFEvalDataset,
        # path='opencompass/IFEval',
        path='/kpfs/dataset/general/ifeval/input_data.jsonl',
        reader_cfg=dict(input_columns=['prompt'], output_column='reference'),
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
            evaluator=dict(type=IFEvaluator),
            pred_role='BOT',
        ),
    )
]

###########################################################################
#  (2) IFBench 数据集配置
#  注意: IFBench 使用 IFBenchEvaluator，与 IFEval 的 IFEvaluator 不同
#         IFBench 对关键词计数等要求精确，小模型容易失分
###########################################################################

ifbench_datasets = [
    dict(
        abbr='IFBench',
        type=IFEvalDataset,
        # path='opencompass/IFbench',
        path='/kpfs/dataset/general/IFBench/IFBench_test.jsonl',
        reader_cfg=dict(input_columns=['prompt'], output_column='reference'),
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
            evaluator=dict(type=IFBenchEvaluator),
            pred_role='BOT',
        ),
    )
]

###########################################################################
#  (2b) Multi-IF 数据集配置 (仅中英文)
#  多轮推理：每轮对话拼接历史后调用模型，忠实于原始 Multi-IF 评测方式
###########################################################################

multiif_datasets = [
    dict(
        abbr='Multi-IF_Chinese',
        type=MultiIFDataset,
        path='/kpfs/dataset/general/multi_if/multiIF_20241018.csv',
        language='Chinese',
        max_turns=3,
        reader_cfg=dict(input_columns=['prompt'], output_column='reference'),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(role='HUMAN', prompt='{prompt}'),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(
                type=MultiIFInferencer,
                max_out_len=32768,
            ),
        ),
        eval_cfg=dict(
            evaluator=dict(type=MultiIFEvaluator),
            pred_role='BOT',
        ),
    ),
    dict(
        abbr='Multi-IF_English',
        type=MultiIFDataset,
        path='/kpfs/dataset/general/multi_if/multiIF_20241018.csv',
        language='English',
        max_turns=3,
        reader_cfg=dict(input_columns=['prompt'], output_column='reference'),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(role='HUMAN', prompt='{prompt}'),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(
                type=MultiIFInferencer,
                max_out_len=32768,
            ),
        ),
        eval_cfg=dict(
            evaluator=dict(type=MultiIFEvaluator),
            pred_role='BOT',
        ),
    ),
]

# 统一设置
for ds_list in [ifeval_datasets, ifbench_datasets, multiif_datasets]:
    for ds in ds_list:
        ds['n'] = 1
        ds['infer_cfg']['inferencer']['max_out_len'] = 32768

datasets = ifeval_datasets + ifbench_datasets + multiif_datasets

###########################################################################
#  (3) 模型列表
###########################################################################

models = []

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
            top_k=20, temperature=0.6, top_p=0.95, do_sample=True
        ),
        chat_template_kwargs=dict(enable_thinking=True),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    ),
]

# ── Qwen3-4B 非思考模式 ──
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
            top_k=20, temperature=0.7, top_p=0.8, do_sample=True
        ),
        chat_template_kwargs=dict(enable_thinking=False),
        run_cfg=dict(num_gpus=1),
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
        batch_size=8,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    ),
]

###########################################################################
#  (4) 推理 & 评测
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
#  (5) 汇总器 & 输出
###########################################################################

summarizer = dict(
    dataset_abbrs=[
        # ── IFEval ──
        ['IFEval', 'Prompt-level-strict-accuracy'],
        ['IFEval', 'Inst-level-strict-accuracy'],
        ['IFEval', 'Prompt-level-loose-accuracy'],
        ['IFEval', 'Inst-level-loose-accuracy'],
        # ── IFBench ──
        ['IFBench', 'score'],
        ['IFBench', 'Prompt-level-strict-accuracy'],
        ['IFBench', 'Inst-level-strict-accuracy'],
        ['IFBench', 'Prompt-level-loose-accuracy'],
        ['IFBench', 'Inst-level-loose-accuracy'],
        # ── Multi-IF 中文 ──
        ['Multi-IF_Chinese', 'score'],
        ['Multi-IF_Chinese', 'Prompt-level-strict-accuracy'],
        ['Multi-IF_Chinese', 'Inst-level-strict-accuracy'],
        ['Multi-IF_Chinese', 'Prompt-level-loose-accuracy'],
        ['Multi-IF_Chinese', 'Inst-level-loose-accuracy'],
        ['Multi-IF_Chinese', 'Chinese_turn1_Prompt-level-strict-accuracy'],
        ['Multi-IF_Chinese', 'Chinese_turn1_Inst-level-strict-accuracy'],
        ['Multi-IF_Chinese', 'Chinese_turn1_Prompt-level-loose-accuracy'],
        ['Multi-IF_Chinese', 'Chinese_turn1_Inst-level-loose-accuracy'],
        ['Multi-IF_Chinese', 'Chinese_turn2_Prompt-level-strict-accuracy'],
        ['Multi-IF_Chinese', 'Chinese_turn2_Inst-level-strict-accuracy'],
        ['Multi-IF_Chinese', 'Chinese_turn2_Prompt-level-loose-accuracy'],
        ['Multi-IF_Chinese', 'Chinese_turn2_Inst-level-loose-accuracy'],
        ['Multi-IF_Chinese', 'Chinese_turn3_Prompt-level-strict-accuracy'],
        ['Multi-IF_Chinese', 'Chinese_turn3_Inst-level-strict-accuracy'],
        ['Multi-IF_Chinese', 'Chinese_turn3_Prompt-level-loose-accuracy'],
        ['Multi-IF_Chinese', 'Chinese_turn3_Inst-level-loose-accuracy'],
        # ── Multi-IF 英文 ──
        ['Multi-IF_English', 'score'],
        ['Multi-IF_English', 'Prompt-level-strict-accuracy'],
        ['Multi-IF_English', 'Inst-level-strict-accuracy'],
        ['Multi-IF_English', 'Prompt-level-loose-accuracy'],
        ['Multi-IF_English', 'Inst-level-loose-accuracy'],
        ['Multi-IF_English', 'English_turn1_Prompt-level-strict-accuracy'],
        ['Multi-IF_English', 'English_turn1_Inst-level-strict-accuracy'],
        ['Multi-IF_English', 'English_turn1_Prompt-level-loose-accuracy'],
        ['Multi-IF_English', 'English_turn1_Inst-level-loose-accuracy'],
        ['Multi-IF_English', 'English_turn2_Prompt-level-strict-accuracy'],
        ['Multi-IF_English', 'English_turn2_Inst-level-strict-accuracy'],
        ['Multi-IF_English', 'English_turn2_Prompt-level-loose-accuracy'],
        ['Multi-IF_English', 'English_turn2_Inst-level-loose-accuracy'],
        ['Multi-IF_English', 'English_turn3_Prompt-level-strict-accuracy'],
        ['Multi-IF_English', 'English_turn3_Inst-level-strict-accuracy'],
        ['Multi-IF_English', 'English_turn3_Prompt-level-loose-accuracy'],
        ['Multi-IF_English', 'English_turn3_Inst-level-loose-accuracy'],
    ],
)

work_dir = './results/instruction_following_vllm_eval'
