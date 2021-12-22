#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function
import os
import json
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from math import log2
import numpy as np
from scipy import stats
import tensorflow as tf

CUSTOMER_NUGGET_TYPES = ('CNUG0', 'CNUG', 'CNUG*', 'CNaN')
HELPDESK_NUGGET_TYPES = ('HNUG', 'HNUG*', 'HNaN')
QUALITY_MEASURES = ("A", "E", "S")
QUALITY_SCALES = ('2', '1', '0', '-1', '-2')


def normalize(pred, truth):
    """ convert inputs to np.array and make sure
    inputs are normalized probability distributions
    """
    if len(pred) != len(truth):
        raise ValueError("pred and truth have different lengths")
    if len(pred) == 0 or len(truth) == 0:
        raise ValueError("pred or truth are empty")

    pred, truth = np.asarray(pred), np.asarray(truth)
    if not ((pred >= 0).all() and (truth >= 0).all()):
        raise ValueError(f"probability distribution should not be negative {pred}")
    pred, truth = pred / pred.sum(), truth / truth.sum()
    return pred, truth


def normalized_match_dist(pred, truth):
    """NMD: Normalized Match Distance"""
    pred, truth = normalize(pred, truth)
    cum_p, cum_q = np.cumsum(pred), np.cumsum(truth)
    return (np.abs(cum_p - cum_q)).sum() / (len(pred) - 1.)


def distance_weighted(pred, truth, i):
    return np.sum([np.abs(i - j) * ((pred[j] - truth[j]) ** 2) for j in range(len(pred))])


def order_aware_div(pred, truth):
    return np.mean([distance_weighted(pred, truth, i) for i in range(len(pred)) if pred[i] > 0])


def rsnod(pred, truth):
    """ RSNOD: Root Symmetric Normalised Order-Aware Divergence
    """

    pred, truth = normalize(pred, truth)
    sod = (order_aware_div(pred, truth) + order_aware_div(truth, pred)) / 2.
    return np.sqrt((sod / (len(pred) - 1)))


def root_normalized_squared_error(pred, truth):
    """ RNSS: Root Normalised Sum of Squares
    """

    def squared_error(pred, truth):
        return ((pred - truth) ** 2).sum()

    pred, truth = normalize(pred, truth)
    return np.sqrt(squared_error(pred, truth) / 2)


def jensen_shannon_div(pred, truth, base=2):
    ''' JSD: Jensen-Shannon Divergence
    '''
    pred, truth = normalize(pred, truth)
    m = 1. / 2 * (pred + truth)
    return (stats.entropy(pred, m, base=base)
            + stats.entropy(truth, m, base=base)) / 2.


def evaluate_nugget(id2pred, id2truth, alpha=.5, strict=True):
    def _evaluate_nugget(measure):
        def _truth2prob(labels, nugget_types):
            c = Counter(labels)
            prob = []
            for nugget_type in nugget_types:
                prob.append(c.get(nugget_type, 0))
            prob = np.array(prob, dtype=np.float64)
            prob /= prob.sum()

            return prob

        def _pred_2_prob(score_dict, nugget_types):
            score_dict = deepcopy(score_dict)
            prob = np.array([score_dict.pop(nugget_type, 0)
                             for nugget_type in nugget_types])
            if score_dict:
                raise ValueError("contain illegal nugget type in prediction")
            return prob

        if strict:
            check_missing_prediction(id2pred, id2truth)

        dialogue_scores = []

        for idx, prediction in id2pred.items():
            truth = id2truth[idx]
            prediction = prediction["nugget"]
            is_customer = [t["sender"] == "customer" for t in truth["turns"]]
            assert len(is_customer) == len(prediction)

            c_turns_scores = []
            h_turns_scores = []

            for i, turn_pred in enumerate(prediction):
                nugget_types = CUSTOMER_NUGGET_TYPES if is_customer[i] else HELPDESK_NUGGET_TYPES
                truth_labels = (anno["nugget"][i] for anno in truth["annotations"])

                truth_prob = _truth2prob(truth_labels, nugget_types)
                score = measure(
                    _pred_2_prob(turn_pred, nugget_types),
                    truth_prob
                )

                if is_customer[i]:
                    c_turns_scores.append(score)
                else:
                    h_turns_scores.append(score)

            dialogue_scores.append(np.mean(c_turns_scores) * alpha + np.mean(h_turns_scores) * (1 - alpha))

        return -log2(np.mean(dialogue_scores))

    return {
        "jsd": _evaluate_nugget(jensen_shannon_div),
        "rnss": _evaluate_nugget(root_normalized_squared_error)
    }


def evaluate_quality(id2pred, id2truth, strict=False):
    def _evaluate_quality(measure):
        def _truth2prob(labels):
            c = Counter(labels)
            prob = []
            for scale in QUALITY_SCALES:
                score = c.pop(scale, 0)
                prob.append(score)
            prob = np.array(prob, dtype=np.float64)
            prob /= prob.sum()
            return prob

        def _pred_2_prob(score_dict):
            score_dict = deepcopy(score_dict)
            prob = np.array(
                [score_dict.pop(scale, 0) for scale in QUALITY_SCALES])
            if score_dict:
                raise ValueError("contain illegal quality scale in prediction")
            return prob

        if strict:
            check_missing_prediction(id2pred, id2truth)

        result = {}
        for idx, prediction in id2pred.items():
            truth = id2truth[idx]
            prediction = prediction["quality"]
            for score_key in prediction:
                truth_labels = (str(anno["quality"][score_key])
                                for anno in truth["annotations"])
                result.setdefault(score_key, [])
                score = measure(
                    _pred_2_prob(prediction[score_key]),
                    _truth2prob(truth_labels))
                result[score_key].append(score)

        for key, value in result.items():
            # use -log2 to make the score more readable.
            result[key] = -log2(np.mean(value))
        return result

    return {
        "rsnod": _evaluate_quality(rsnod),
        "nmd": _evaluate_quality(normalized_match_dist)
    }


# dev_inputs = (dialogue_id, [input_ids, input_mask, input_type_ids, dialogue_length, turn_number])

def nugget_pred_to_dict(customer_prob, helpdesk_prob, dialogue_length):
    customer_length = (dialogue_length // 2) + 1 if dialogue_length % 2 == 1 else dialogue_length // 2
    helpdesk_length = dialogue_length // 2

    customer_prob = customer_prob[:customer_length]
    helpdesk_prob = helpdesk_prob[:helpdesk_length]

    nugget_list = []

    # 0 1 2 3 4 5
    for i in range(dialogue_length):
        result_dict = {}
        if i % 2 == 0:
            nugget_type = CUSTOMER_NUGGET_TYPES
            distribution = customer_prob[i // 2]

        else:
            nugget_type = HELPDESK_NUGGET_TYPES
            distribution = helpdesk_prob[i // 2]

        for nugget_type, prob in zip(nugget_type, distribution):
            result_dict[nugget_type] = float(prob)

        nugget_list.append(result_dict)

    return nugget_list


def quality_pred_to_dict(quality_probs):
    result = {}
    for measure, quality_prob in zip(QUALITY_MEASURES, quality_probs):
        result[str(measure)] = {}
        for scale, prob in zip(QUALITY_SCALES, quality_prob):
            result[str(measure)][str(scale)] = float(prob)

    return result


def pred_to_submission(inputs, model, task, output_file, write_to_file=True):
    # model_input = (dialogue_id, [input_ids, input_mask, input_type_ids, sentence_ids, sentence_mask,
    # customer_labels, helpdesk_labels])
    submission = []
    count = 0
    for dialogue in inputs:
        dialogue_id = dialogue[0]
        if task == "nugget":
            model_input = dialogue[1]
            sentence_mask = dialogue[1][4]
            dialogue_length = np.squeeze(tf.math.count_nonzero(sentence_mask, axis=-1).numpy(), axis=-1)

            # pred
            customer_prob, helpdesk_prob = model.predict(x=model_input)
            customer_prob = tf.squeeze(customer_prob, axis=0)
            helpdesk_prob = tf.squeeze(helpdesk_prob, axis=0)
            nugget_list = nugget_pred_to_dict(customer_prob, helpdesk_prob, dialogue_length)
            submission_format = {
                "id": dialogue_id,
                "nugget": nugget_list  # a list of dict
            }
            submission.append(submission_format)
            count += 1
            print("Predicting ", count, " / 300 dialogues...")

        else:
            model_input = dialogue[1]
            quality_probs = model.predict(x=model_input)
            result = quality_pred_to_dict(quality_probs)
            submission_format = {
                "id": dialogue_id,
                "quality": result
            }
            submission.append(submission_format)
            count += 1
            print("Predicting ", count, " / 300 dialogues...")

    if write_to_file:
        with open(output_file, 'w') as f:
            json.dump(submission, f)

    return submission


def check_missing_prediction(id2pred, id2truth):
    for dialogue_id in id2truth:
        if dialogue_id not in id2pred:
            raise ValueError("Missing prediction for dialogue id %s" % dialogue_id)


def evaluate_from_list(task, pred, truth, alpha, strict):
    if not pred:
        raise ValueError("Prediction JSON is empty")
    if not truth:
        raise ValueError("Ground truth JSON is empty")

    id2pred = {d["id"]: d for d in pred}
    id2truth = {d["id"]: d for d in truth}
    results = {}
    if task == "nugget":
        results = evaluate_nugget(id2pred=id2pred, id2truth=id2truth, alpha=alpha, strict=strict)

    else:
        results = evaluate_quality(id2pred=id2pred, id2truth=id2truth, strict=strict)

    print(results)
    return results


def evaluate(task, pred_path, truth_path, alpha=.5, strict=False):
    pred = json.load(open(pred_path, encoding="utf-8"))
    truth = json.load(open(truth_path, encoding="utf-8"))

    return evaluate_from_list(task, pred, truth, alpha, strict)
