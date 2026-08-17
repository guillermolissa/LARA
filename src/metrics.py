import pandas as pd
import math
import numpy as np
from typing import List

def apk(ground_truth: list, predicted: list,  k=5, ignore_item=['[PAD]']) -> float:
    """
    Computes the average precision at k.
    ----------
    
    ground_truth : list
        A list of actual items to be predicted
    
    predicted : list
        An ordered list of predicted items
    
    k : int, default = 5
        Number of predictions from the list predicted to consider
    Returns:
    -------
    score : float
        The average precision at k.
    """
    
    def _precision(predicted, ground_truth):
        # Filter ignored items from both lists before computing precision
        predicted_filtered = [v for v in predicted if v not in ignore_item]
        hits = [v for v in predicted_filtered if v in ground_truth]
        if not predicted_filtered:
            return 0.0
        return float(len(hits)) / float(len(predicted_filtered))

    if not predicted or not ground_truth:
        return 0.0

    predicted = [item for item in predicted if item not in ignore_item]
    ground_truth = [item for item in ground_truth if item not in ignore_item]

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0

    for i, p in enumerate(predicted):
        if p in ground_truth and p not in predicted[:i]:
            max_ix = min(i + 1, len(predicted))
            score += _precision(predicted[:max_ix], ground_truth)

    if score == 0.0:
        return 0.0

    # Normalize by min(|ground_truth|, k) — standard AP@k definition
    return score / min(len(ground_truth), k)


# def mapk(ground_truth: List[list], predicted: List[list], k: int=10) -> float:
#     """
#     Computes the mean average precision at k.
#     Parameters
#     ----------
#     ground_truth : a list of lists
#         Actual items to be predicted
#         example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
#     predicted : a list of lists
#         Ordered predictions
#         example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
#     Returns:
#     -------
#         mark: float
#             The mean average precision at k (map@k)
#     """
#     if len(ground_truth) != len(predicted):
#         raise AssertionError("Length mismatched")
    
#     return np.mean([apk(a,p,k) for a,p in zip(ground_truth, predicted)])

# def dcg_at_k(r, k, method=0):
#     """Score is discounted cumulative gain (dcg)
#     Relevance is positive real values.  Can use binary
#     as the previous methods.
#     Example from
#     http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
#     >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
#     >>> dcg_at_k(r, 1)
#     3.0
#     >>> dcg_at_k(r, 1, method=1)
#     3.0
#     >>> dcg_at_k(r, 2)
#     5.0
#     >>> dcg_at_k(r, 2, method=1)
#     4.2618595071429155
#     >>> dcg_at_k(r, 10)
#     9.6051177391888114
#     >>> dcg_at_k(r, 11)
#     9.6051177391888114
#     Args:
#         r: Relevance scores (list or numpy) in rank order
#             (first element is the first item)
#         k: Number of results to consider
#         method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
#                 If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
#     Returns:
#         Discounted cumulative gain
#     """
#     r = np.asfarray(r)[:k]
#     if r.size:
#         if method == 0:
#             return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
#         elif method == 1:
#             return np.sum(r / np.log2(np.arange(2, r.size + 2)))
#         else:
#             raise ValueError('method must be 0 or 1.')
#     return 0.


# def ndcg_at_k(r, k, method=0):
#     """Score is normalized discounted cumulative gain (ndcg)
#     Relevance is positive real values.  Can use binary
#     as the previous methods.
#     Example from
#     http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
#     >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
#     >>> ndcg_at_k(r, 1)
#     1.0
#     >>> r = [2, 1, 2, 0]
#     >>> ndcg_at_k(r, 4)
#     0.9203032077642922
#     >>> ndcg_at_k(r, 4, method=1)
#     0.96519546960144276
#     >>> ndcg_at_k([0], 1)
#     0.0
#     >>> ndcg_at_k([1], 2)
#     1.0
#     Args:
#         r: Relevance scores (list or numpy) in rank order
#             (first element is the first item)
#         k: Number of results to consider
#         method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
#                 If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
#     Returns:
#         Normalized discounted cumulative gain
#     """
#     dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
#     if not dcg_max:
#         return 0.
#     return dcg_at_k(r, k, method) / dcg_max


def precision_k(ground_truth: list, predicted: list,  k=5, ignore_item=['[PAD]']) -> float:
    """
        Computes the precision at K.
        ----------
        ground_truth : list
            A list of actual items to be predicted
            
        predicted : list
            An ordered list of predicted items

        
        k : int, default = 5
            Number of predictions from the list predicted to consider
        
        ignore_item : list, default = ['[PAD]']
            Items to ignore in the evaluation.
        Returns:
        -------
        score : float
            The precision at k.
    """

    predicted = [item for item in predicted if item not in ignore_item]
    ground_truth = [item for item in ground_truth if item not in ignore_item]

    ground_truth_set = set(ground_truth)
    hits = sum(1 for item in predicted[:k] if item in ground_truth_set)
    return hits / k if k > 0 else 0.0



def recall_k(ground_truth: list, predicted: list,  k=5, ignore_item=['[PAD]']) -> float:
    """
        Computes the recall at K.
        ----------
        ground_truth : list
            A list of actual items to be predicted
            
        predicted : list
            An ordered list of predicted items

        
        k : int, default = 5
            Number of predictions from the list predicted to consider
        
        ignore_item : list, default = ['[PAD]']
            Items to ignore in the evaluation.
        
        Returns:
        -------
        score : float
            The precision at k.
    """
    predicted = [item for item in predicted if item not in ignore_item]
    ground_truth = [item for item in ground_truth if item not in ignore_item]
    
    predicted_at_k = set(predicted[:k])  # Top K predicted items
    relevant_items = set(ground_truth)  # Ground truth items
    relevant_at_k = predicted_at_k & relevant_items  # Intersection
    
    if not relevant_items:  # Avoid division by zero
        return 0.0
    
    return len(relevant_at_k) / len(relevant_items)



def hit_rate_k(ground_truth: list, predicted: list,  k=5, ignore_item=['[PAD]']) -> int:
    """
        Calculates the Hit Rate at a cutoff K.
        ----------
        ground_truth : list
            A list of actual items to be predicted
            
        predicted : list
            An ordered list of predicted items

        
        k : int, default = 5
            The cutoff point for the ranking.
        
        ignore_item : list, default = ['[PAD]']
            Items to ignore in the evaluation.
        Returns:
        -------
        score : float
            The hit rate at k.
    """
    
    predicted = [item for item in predicted if item not in ignore_item]
    ground_truth = [item for item in ground_truth if item not in ignore_item]
    
    predicted_at_k = set(predicted[:k])  # Top K predictions
    relevant_items = set(ground_truth)  # Ground truth items
    return 1 if predicted_at_k & relevant_items else 0  # Hit if there is any overlap


def rr_k(ground_truth: list, predicted: list,  k=5, ignore_item=['[PAD]']) -> float:
    """
        Calculates the Reciprocal Rank for Mean Reciprocal Rank (MRR) at a cutoff K.
        ----------
        ground_truth : list
            A list of actual items to be predicted
            
        predicted : list
            An ordered list of predicted items

        
        k : int, default = 5
            The cutoff point for the ranking.
            
        ignore_item : list, default = ['[PAD]']
            Items to ignore in the evaluation.
        Returns:
        -------
        score : float
            The RR at k.
    """
    
    total_reciprocal_rank = 0.0
    predicted = [item for item in predicted if item not in ignore_item]
    ground_truth = [item for item in ground_truth if item not in ignore_item]

    pred_list = predicted[:k]  # Truncate to top K
    relevant_items = set(ground_truth)

        # Find the rank of the first relevant item
    for rank, item in enumerate(pred_list, 1):
        if item in relevant_items:
            total_reciprocal_rank = 1.0 / rank
            break  # Stop at the first relevant item

    return total_reciprocal_rank


def dcg(relevances, k=None):
    if k is None:
        k = len(relevances)
    relevances = np.asarray(relevances)[:k]
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum((2**relevances - 1) / discounts)


def ndcg_k(ideal_list, proposal_list, k=None, ignore_item=['[PAD]']) -> float:

    ideal_list = [item for item in ideal_list if item not in ignore_item]
    proposal_list = [item for item in proposal_list if item not in ignore_item]

    # When k is None, use len(ideal_list) so DCG and IDCG are computed over
    # the same number of positions and NDCG is bounded in [0, 1].
    if k is None:
        k = len(ideal_list)

    relevance_dict = {item: 1 for item in ideal_list}
    proposal_rels = [relevance_dict.get(item, 0) for item in proposal_list[:k]]
    ideal_rels = sorted(relevance_dict.values(), reverse=True)
    dcg_val = dcg(proposal_rels, k)
    idcg_val = dcg(ideal_rels, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0
