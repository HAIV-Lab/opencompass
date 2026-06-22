from opencompass.datasets import MMLUReduxDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator,MMLUReduxAccEvaluator
from opencompass.utils.text_postprocessors import match_answer_pattern



# MMLU-Redux 包含 57 个子集
mmlu_redux_subsets = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics',
    'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
    'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning',
    'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes',
    'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
    'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations',
    'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]

QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{input}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

mmlu_redux_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    test_split='test',
    train_split='test'
    # test_range='[:3]' 
)

mmlu_redux_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=QUERY_TEMPLATE),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mmlu_redux_eval_cfg = dict(
    evaluator=dict(type=MMLUReduxAccEvaluator),
    pred_postprocessor=dict(type=match_answer_pattern, answer_pattern=r'(?i)ANSWER\s*:\s*([A-D])'))

# print(f"==============read dataset")
# 为每个子集生成配置
mmlu_redux_datasets = []
for subset in mmlu_redux_subsets:
    mmlu_redux_datasets.append(
        dict(
            type=MMLUReduxDataset,
            abbr=f'mmlu_redux_{subset}',
            path='opencompass/mmlu_redux',  # 数据集标识符. 与dataset_info中配置保持一致
            name=subset,  # 子集名称
            reader_cfg=mmlu_redux_reader_cfg,
            infer_cfg=mmlu_redux_infer_cfg,
            eval_cfg=mmlu_redux_eval_cfg,
        )
    )