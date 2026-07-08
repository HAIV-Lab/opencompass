import csv
import json
import os
from typing import Optional

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_inferencer.icl_gen_inferencer import GenInferencer
from opencompass.openicl.icl_inferencer.icl_base_inferencer import \
    dump_results_dict
from opencompass.registry import ICL_INFERENCERS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from ..IFEval.evaluation_main import (InputExample,
                                      test_instruction_following_loose,
                                      test_instruction_following_strict)

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None

from ..base import BaseDataset
from ..IFEval.evaluation_main import (InputExample,
                                      test_instruction_following_loose,
                                      test_instruction_following_strict)

##############################################################################
#  Dataset — 保持完整多轮记录
##############################################################################


@LOAD_DATASET.register_module()
class MultiIFDataset(BaseDataset):
    """Multi-IF dataset — each sample is a full multi-turn conversation."""

    @staticmethod
    def load(path, language=None, max_turns=3):
        path = get_data_path(path)
        ext = os.path.splitext(path)[1].lower()

        rows = []
        if ext == '.csv':
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))

        records = []
        for row in rows:
            if language is not None and row.get('language') != language:
                continue
            raw_prompt = row.get('turn_1_prompt', '')
            first_prompt = ''
            if raw_prompt:
                try:
                    first_prompt = json.loads(raw_prompt).get('content', '')
                except json.JSONDecodeError:
                    first_prompt = raw_prompt
            records.append({
                'prompt': first_prompt,
                'reference': row,
            })
        return Dataset.from_list(records)


##############################################################################
#  Inferencer — 多轮推理
##############################################################################


class MultiIFOutputHandler:

    def __init__(self):
        self.results_dict = {}

    def save_results(self, origin_prompt, prediction, idx, gold=None):
        self.results_dict[str(idx)] = {
            'origin_prompt': origin_prompt,
            'prediction': prediction,
            'gold': gold,
        }

    def write_to_json(self, save_dir, filename):
        dump_results_dict(self.results_dict, os.path.join(save_dir, filename))


@ICL_INFERENCERS.register_module()
class MultiIFInferencer(GenInferencer):
    """Multi-turn inferencer for Multi-IF."""

    def __init__(self,
                 model,
                 max_out_len: int = 32768,
                 max_seq_len: Optional[int] = None,
                 batch_size: int = 1,
                 output_json_filepath: str = './icl_inference_output',
                 output_json_filename: str = 'predictions',
                 save_every: int = 1,
                 **kwargs):
        super().__init__(
            model=model,
            max_out_len=max_out_len,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            save_every=save_every,
            **kwargs,
        )

    def inference(self,
                  retriever,
                  ice_template=None,
                  prompt_template=None,
                  output_json_filepath=None,
                  output_json_filename=None):
        output_handler = MultiIFOutputHandler()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        ice_idx_list = retriever.retrieve()
        prompt_list = self.get_generation_prompt_list_from_retriever_indices(
            ice_idx_list,
            retriever,
            gen_field_replace_token='',
            max_seq_len=self.max_seq_len,
            ice_template=ice_template,
            prompt_template=prompt_template)

        ds_reader = retriever.dataset_reader
        gold_ans = ds_reader.dataset['test'][ds_reader.output_column]
        prompt_list = list(zip(prompt_list, gold_ans))

        import logging
        from tqdm import tqdm

        os.makedirs(output_json_filepath, exist_ok=True)
        logger = logging.getLogger(__name__)
        model = self.model

        for idx, (prompt, record) in enumerate(tqdm(prompt_list, desc='Multi-IF')):
            max_turns = next(
                (s for s in [3, 2, 1] if record.get(f'turn_{s}_prompt')), 1)

            conversation = []
            turn_predictions = []

            for step in range(1, max_turns + 1):
                raw = record.get(f'turn_{step}_prompt')
                if not raw:
                    break
                try:
                    turn_prompt_data = json.loads(raw)
                except json.JSONDecodeError:
                    break
                user_content = turn_prompt_data.get('content', '')
                if not user_content:
                    break

                conversation.append({'role': 'user', 'content': user_content})

                # ── 绕过 generate_from_template，直接格式化并调用 vLLM ──
                try:
                    # 1. apply_chat_template → 格式化对话为字符串
                    formatted = model.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                        **model.chat_template_kwargs,
                    )
                    # 2. 构造 SamplingParams
                    stop_words = list(set(model.stop_words))
                    sampling_kwargs = SamplingParams(
                        temperature=model.generation_kwargs.get('temperature', 0),
                        max_tokens=self.max_out_len,
                        stop=stop_words,
                    )
                    # 3. 直接调用 vLLM generate
                    outputs = model.model.generate(
                        [formatted], sampling_kwargs)
                    response = outputs[0].outputs[0].text
                except Exception as e:
                    logger.error(f'Sample {idx} turn {step} failed: {e}')
                    response = ''

                conversation.append({'role': 'assistant', 'content': response})
                turn_predictions.append(response)

            output_handler.save_results(
                origin_prompt=prompt,
                prediction=turn_predictions,
                idx=idx,
                gold=record,
            )

        output_handler.write_to_json(output_json_filepath, output_json_filename)
        return output_handler.results_dict


##############################################################################
#  Evaluator
##############################################################################


class MultiIFEvaluator(BaseEvaluator):

    def score(self, predictions, references, origin_prompt):
        prompt_strict_correct = prompt_strict_total = 0
        inst_strict_correct = inst_strict_total = 0
        prompt_loose_correct = prompt_loose_total = 0
        inst_loose_correct = inst_loose_total = 0
        lang_turn_results = {}
        details = {}

        def _parse_step(refer, step):
            try:
                ids = json.loads(refer.get(f'turn_{step}_instruction_id_list', '[]'))
            except Exception:
                ids = []
            try:
                kwargs_raw = json.loads(refer.get(f'turn_{step}_kwargs', '[]'))
            except Exception:
                kwargs_raw = []
            kwargs = []
            for k in kwargs_raw:
                try:
                    kwargs.append(json.loads(k) if isinstance(k, str) else k)
                except Exception:
                    kwargs.append({})
            return ids, kwargs

        for index, (pred_list, refer) in enumerate(zip(predictions, references)):
            if not isinstance(pred_list, list):
                pred_list = [pred_list]
            language = refer.get('language', 'unknown')

            for turn_idx, response in enumerate(pred_list):
                step = turn_idx + 1
                lt_key = f'{language}_turn{step}'
                inst_ids, kwargs = _parse_step(refer, step)

                inp = InputExample(
                    key=refer.get('key', index),
                    instruction_id_list=inst_ids,
                    prompt=origin_prompt[index] if origin_prompt else '',
                    kwargs=kwargs,
                )
                for kwarg in inp.kwargs:
                    for k in list(kwarg.keys()):
                        if kwarg[k] is None:
                            kwarg.pop(k, None)

                ex = test_instruction_following_strict(inp, response)
                is_strict = all(ex.follow_instruction_list)
                prompt_strict_correct += is_strict
                prompt_strict_total += 1
                inst_strict_correct += sum(ex.follow_instruction_list)
                inst_strict_total += len(ex.instruction_id_list)

                ex2 = test_instruction_following_loose(inp, response)
                is_loose = all(ex2.follow_instruction_list)
                prompt_loose_correct += is_loose
                prompt_loose_total += 1
                inst_loose_correct += sum(ex2.follow_instruction_list)
                inst_loose_total += len(ex2.instruction_id_list)

                lt = lang_turn_results.setdefault(lt_key, {
                    'ps_c': 0, 'ps_t': 0, 'is_c': 0, 'is_t': 0,
                    'pl_c': 0, 'pl_t': 0, 'il_c': 0, 'il_t': 0,
                })
                lt['ps_c'] += is_strict
                lt['ps_t'] += 1
                lt['is_c'] += sum(ex.follow_instruction_list)
                lt['is_t'] += len(ex.instruction_id_list)
                lt['pl_c'] += is_loose
                lt['pl_t'] += 1
                lt['il_c'] += sum(ex2.follow_instruction_list)
                lt['il_t'] += len(ex2.instruction_id_list)

                grade = 'strict' if is_strict else ('loose' if is_loose else 'none')
                details[f'{index}_turn{step}'] = {
                    'pred': response,
                    'language': language,
                    'turn': step,
                    'is_strict_correct': is_strict,
                    'is_loose_correct': is_loose,
                    'grade': grade,
                }

        def _d(a, b):
            return a / b * 100 if b > 0 else 0.0

        results = {
            'score': (_d(prompt_strict_correct, prompt_strict_total) +
                      _d(inst_strict_correct, inst_strict_total) +
                      _d(prompt_loose_correct, prompt_loose_total) +
                      _d(inst_loose_correct, inst_loose_total)) / 4,
            'Prompt-level-strict-accuracy': _d(prompt_strict_correct, prompt_strict_total),
            'Inst-level-strict-accuracy': _d(inst_strict_correct, inst_strict_total),
            'Prompt-level-loose-accuracy': _d(prompt_loose_correct, prompt_loose_total),
            'Inst-level-loose-accuracy': _d(inst_loose_correct, inst_loose_total),
            'details': details,
        }
        for lt_key, lt in lang_turn_results.items():
            results[f'{lt_key}_Prompt-level-strict-accuracy'] = _d(lt['ps_c'], lt['ps_t'])
            results[f'{lt_key}_Inst-level-strict-accuracy'] = _d(lt['is_c'], lt['is_t'])
            results[f'{lt_key}_Prompt-level-loose-accuracy'] = _d(lt['pl_c'], lt['pl_t'])
            results[f'{lt_key}_Inst-level-loose-accuracy'] = _d(lt['il_c'], lt['il_t'])

        return results
