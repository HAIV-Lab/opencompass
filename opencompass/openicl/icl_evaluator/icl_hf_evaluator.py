import os
import random
from typing import List, Optional
import ast
import re
import evaluate
import numpy as np
from datasets import Dataset
from mmengine.config import ConfigDict

from opencompass.registry import ICL_EVALUATORS

from .icl_base_evaluator import BaseEvaluator


class HuggingfaceEvaluator(BaseEvaluator):
    """Use huggingface evaluate module to calculate the target metrics.

    Args:
        metric (str): Metric name in evaluate module.
        seed (int): There exists some randomness during the calculation of some
            metrics, thus we set a fixed random seed for reproducing. Defaults
            to 0.
        pred_postprocessor (optional): Function or configuration for prediction
            post-processing.
    """

    def __init__(self,
                 metric: str,
                 seed: int = 0,
                 pred_postprocessor=None) -> None:
        self.metric = metric
        self.seed = seed
        super().__init__(pred_postprocessor=pred_postprocessor)

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
        return {
            'predictions': predictions,
            'references': references,
        }

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        return scores

    def score(self,
              predictions: List,
              references: List,
              test_set=None) -> dict:
        """Calculate scores.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: calculated scores.
        """
        random_state = random.getstate()
        np_random_state = np.random.get_state()

        random.seed(self.seed)
        np.random.seed(self.seed)
        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }
        # use codes pre-downloaded to opencompass repo, avoid downloading
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'hf_metrics', self.metric + '.py')
        if os.path.exists(local_path):
            metric = evaluate.load(local_path)
        else:
            metric = evaluate.load(self.metric)
        scores = metric.compute(**self._preprocess(predictions, references))
        result = self._postprocess(scores)
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        return result


@ICL_EVALUATORS.register_module()
class AccEvaluator(HuggingfaceEvaluator):
    """Accuracy evaluator."""

    def __init__(self,
                 pred_postprocessor: Optional[ConfigDict] = None) -> None:
        super().__init__(metric='accuracy',
                         pred_postprocessor=pred_postprocessor)

    def _preprocess(self,
                    predictions: List,
                    references: List,
                    test_set=None) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
        print(f"=====references:{references}=======")
        print(f"=====predictions{predictions}=======")
        mapping_to_int_dict = {
            label: idx
            for idx, label in enumerate(set(map(str, references)))
        }
        pred_set = set(predictions)
        for pred in pred_set:
            if str(pred) not in mapping_to_int_dict.keys():
                mapping_to_int_dict[str(pred)] = len(mapping_to_int_dict)
        golds = [mapping_to_int_dict[str(gold)] for gold in references]
        preds = [mapping_to_int_dict[str(pred)] for pred in predictions]
        return {
            'predictions': preds,
            'references': golds,
        }

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        scores['accuracy'] *= 100
        return scores

@ICL_EVALUATORS.register_module()
class MMLUReduxAccEvaluator(HuggingfaceEvaluator):
    """Accuracy evaluator."""

    def __init__(self,
                 pred_postprocessor: Optional[ConfigDict] = None) -> None:
        super().__init__(metric='accuracy',
                         pred_postprocessor=pred_postprocessor)

    def _clean_label(self, label) -> str:
        """统一将标签转换为干净的字符串格式。
        
        支持处理: 
        - "A" -> "A"
        - "['A']" -> "A"
        - ['A'] -> "A"
        - "['A', 'B']" -> "A,B" (多选情况)
        """
        if label is None:
            return ""
        
        label_str = str(label).strip()
        
        # 如果字符串看起来像一个 Python 列表 (例如 "['A']" 或 "['A', 'B']")
        if label_str.startswith('[') and label_str.endswith(']'):
            try:
                # 使用 ast.literal_eval 安全地解析成真正的 Python 列表
                parsed_list = ast.literal_eval(label_str)
                if isinstance(parsed_list, list):
                    # 将列表排序并用逗号拼接，如 ['B', 'A'] -> "A,B" 确保顺序一致
                    return ",".join(sorted([str(x).strip() for x in parsed_list]))
            except (ValueError, SyntaxError):
                # 如果解析失败，保持原样
                pass
        
        # 如果本来就是 Python 列表
        if isinstance(label, list):
            return ",".join(sorted([str(x).strip() for x in label]))
            
        return label_str
    def _preprocess(self,
                    predictions: List,
                    references: List,
                    test_set=None) -> dict:
        # 1. 统一清洗和转换 prediction 和 reference
        cleaned_refs = [self._clean_label(ref) for ref in references]
        cleaned_preds = [self._clean_label(pred) for pred in predictions]
        
        print(f"===== 原 references: {references} ===> 清洗后: {cleaned_refs} =======")
        print(f"===== 原 predictions: {predictions} ===> 清洗后: {cleaned_preds} =======")

        # 2. 构建类别到数字 ID 的映射字典
        # 基础字典由 references 中所有出现过的标签组成
        mapping_to_int_dict = {
            label: idx
            for idx, label in enumerate(sorted(list(set(cleaned_refs))))
        }
        
        # 如果模型预测出了不在 references 里的新标签，也为它分配一个 ID
        for pred in set(cleaned_preds):
            if pred not in mapping_to_int_dict:
                mapping_to_int_dict[pred] = len(mapping_to_int_dict)
                
        # 3. 将文本标签序列转化为数字 ID 序列
        golds = [mapping_to_int_dict[gold] for gold in cleaned_refs]
        preds = [mapping_to_int_dict[pred] for pred in cleaned_preds]
        
        return {
            'predictions': preds,
            'references': golds,
        }

    
    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        scores['accuracy'] *= 100
        return scores
@ICL_EVALUATORS.register_module()
class AccContaminationEvaluator(AccEvaluator):
    """Accuracy evaluator."""

    def score(self, predictions: List, references: List,
              test_set: Dataset) -> dict:
        # group the predictions and references by their contamination status
        clean_predictions, clean_references = [], []
        input_contaminated_predictions, input_contaminated_references = [], []
        input_and_label_contaminated_predictions, \
            input_and_label_contaminated_references = [], []
        for pred, ref, is_clean in zip(predictions, references,
                                       test_set['is_clean']):
            if is_clean == 'clean':
                clean_predictions.append(pred)
                clean_references.append(ref)
            elif is_clean == 'input contamination':
                input_contaminated_predictions.append(pred)
                input_contaminated_references.append(ref)
            elif is_clean == 'input-and-label contamination':
                input_and_label_contaminated_predictions.append(pred)
                input_and_label_contaminated_references.append(ref)
        clean_results = super().score(clean_predictions, clean_references)
        input_contaminated_results = super().score(
            input_contaminated_predictions, input_contaminated_references)
        input_and_label_contaminated_results = super().score(
            input_and_label_contaminated_predictions,
            input_and_label_contaminated_references)

        # rename the keys of the results, add 'clean, 'input contaminated',
        # 'input-and-label contaminated' as prefixes
        clean_results = {f'{k} - clean': v for k, v in clean_results.items()}
        input_contaminated_results = {
            f'{k} - input contaminated': v
            for k, v in input_contaminated_results.items()
        }
        input_and_label_contaminated_results = {
            f'{k} - input-and-label contaminated': v
            for k, v in input_and_label_contaminated_results.items()
        }
        return {
            **clean_results,
            **input_contaminated_results,
            **input_and_label_contaminated_results
        }


@ICL_EVALUATORS.register_module()
class RougeEvaluator(HuggingfaceEvaluator):
    """Rouge evaluator.

    Note: this evaluator is not suitable for chinese datasets.
    """

    def __init__(self,
                 pred_postprocessor: Optional[ConfigDict] = None) -> None:
        super().__init__(metric='rouge', pred_postprocessor=pred_postprocessor)

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        return {k: v * 100 for k, v in scores.items()}


@ICL_EVALUATORS.register_module()
class BleuEvaluator(HuggingfaceEvaluator):
    """Bleu evaluator."""

    def __init__(self,
                 pred_postprocessor: Optional[ConfigDict] = None) -> None:
        super().__init__(metric='sacrebleu',
                         pred_postprocessor=pred_postprocessor)


class BleuFloresEvaluator(HuggingfaceEvaluator):
    """Bleu evaluator using flores200 tokenize."""

    def __init__(self) -> None:
        super().__init__(metric='sacrebleu')

    def _preprocess(self, predictions: List, references: List) -> dict:
        return {
            'predictions': predictions,
            'references': references,
            'tokenize': 'flores200',
        }


@ICL_EVALUATORS.register_module()
class MccEvaluator(AccEvaluator):
    """Matthews correlation evaluator."""

    def __init__(self) -> None:
        super(AccEvaluator, self).__init__(metric='matthews_correlation')

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        scores['matthews_correlation'] *= 100
        return scores


@ICL_EVALUATORS.register_module()
class SquadEvaluator(HuggingfaceEvaluator):
    """Squad evaluator."""

    def __init__(self) -> None:
        super().__init__(metric='squad')

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
        p_list = [{
            'prediction_text': pred.split('\n')[0],
            'id': str(i)
        } for i, pred in enumerate(predictions)]
        r_list = [{
            'answers': {
                'answer_start': [0],
                'text': [ref]
            },
            'id': str(i)
        } for i, ref in enumerate(references)]
        return {
            'predictions': p_list,
            'references': r_list,
        }

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        return scores['f1']


@ICL_EVALUATORS.register_module()
class EDAccEvaluator(AccEvaluator):
    """Edit distance based accuracy evaluator.

    This implementation requires the un-postprocessed outputs from the model,
    and the reference list where each item is structured as:

    .. code-block:: python

        {
            'candidates': [],  # a list of informative answer candidates
            'label': 0,  # the index of the gold answer
        }

    It always matches the model's output to a valid answer with the citerion
    as the minimum editing distance.
    """

    def __init__(self) -> None:
        super().__init__()
        from rapidfuzz.distance import Levenshtein
        self.dist = Levenshtein.distance

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """

        preds = []
        golds = []

        for i in range(len(predictions)):
            pred, ref = predictions[i], references[i]
            dists = []
            for cands in ref['candidates']:
                if isinstance(cands, str):
                    d = self.dist(pred, cands)
                else:
                    d = np.min([self.dist(pred, cand) for cand in cands])
                dists.append(d)
            preds.append(np.argmin(dists))
            golds.append(ref['label'])

        return {
            'predictions': preds,
            'references': golds,
        }


@ICL_EVALUATORS.register_module()
class AccwithDetailsEvaluator(BaseEvaluator):

    def score(self, predictions, references, origin_prompt) -> dict:

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length.'}

        details = {}
        correct, total = 0, 0
        for index, (pred, ref) in enumerate(zip(predictions, references)):
            is_correct = pred == ref
            correct += is_correct
            details[str(index)] = {
                'prompt': origin_prompt[index],
                'pred': pred,
                'refr': ref,
                'is_correct': is_correct,
            }
            total += 1

        results = {'accuracy': correct / total * 100, 'details': details}

        return results
