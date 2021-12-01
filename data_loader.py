import numpy as np
import os, logging
import json
from preprocess import Processor
import tensorflow as tf

CUSTOMER_NUGGET_TYPES_WITH_PAD = ('CNUG0', 'CNUG', 'CNUG*', 'CNaN', 'PAD')
HELPDESK_NUGGET_TYPES_WITH_PAD = ('HNUG', 'HNUG*', 'HNaN', 'PAD')


def transfer_example_to_input(example, is_test=False):
    dataset_dict = {
        "dialogue_id": None,
        "input_ids": None,
        "input_mask": None,
        "input_type_ids": None,
        "sentence_ids": None,
        "customer_labels": None,
        "helpdesk_labels": None,
        "quality_labels": None
    }

    inputs = []

    for key in dataset_dict:
        dataset_dict[key] = getattr(example, key)
        inputs.append(dataset_dict[key])

    if not is_test:
        # id isnt required
        inputs = inputs[1:]

    return inputs


def create_inputs(json_path, plm, max_len, is_train, language):
    if is_train:
        mode = "train"
        processor = Processor(plm=plm, max_len=max_len, language=language)
        raw_data = json.load(open(json_path))
        input_ids = []
        input_mask = []
        input_type_ids = []
        sentence_ids = []
        customer_labels = []
        helpdesk_labels = []
        quality_labels = []
        for dialogue in raw_data:
            example = processor.process_dialogue(dialogue, mode="train", task="nugget")
            data = transfer_example_to_input(example, is_test=False)
            input_ids.append(data[0])
            input_mask.append(data[1])
            input_type_ids.append(data[2])
            sentence_ids.append(data[3])
            customer_labels.append(data[4])
            helpdesk_labels.append(data[5])
            quality_labels.append(data[6])

        model_input = {"input_ids": input_ids,
                       "input_mask": input_mask,
                       "input_type_ids": input_type_ids,
                       "sentence_ids": sentence_ids}

        return model_input, customer_labels, helpdesk_labels, quality_labels, len(processor.tokenizer)


def _pad_sequence(data, pad_id, width=-1):
    if width == -1:
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data, width


def _pad_labels(data, pad_label, width=-1):
    if width == -1:
        width = max(len(d) for d in data)

    pad_labels = []

    for label in data:
        padding_length = width - len(label)
        # print(label)

        new_label = label + list(pad_label for _ in range(padding_length))
        # print(new_label)

        pad_labels.append(new_label)

    return pad_labels, width


def create_dataset(inputs, task, shuffle_buffer_size=10000, batch_size=32):
    pre_sentence_ids = inputs[0]["sentence_ids"]
    pre_customer_labels = inputs[1]
    pre_helpdesk_labels = inputs[2]
    pre_quality_labels = inputs[3]

    # ----PADDING----

    sentence_ids, max_turn_number = _pad_sequence(pre_sentence_ids, pad_id=-1)
    sentence_mask = []

    for x in sentence_ids:
        mask = [0] * max_turn_number
        for i in range(max_turn_number):
            if x[i] != -1:
                mask[i] = 1
        for j in range(max_turn_number):
            if mask[j] == 0:
                x[j] = 0
        sentence_mask.append(mask)

    customer_label_pad = np.array([0., 0., 0., 0., 1.])
    helpdesk_label_pad = np.array([0., 0., 0., 1.])
    quality_label_pad = []

    customer_labels, customer_turns = _pad_labels(pre_customer_labels, customer_label_pad)
    helpdesk_labels, helpdesk_turns = _pad_labels(pre_helpdesk_labels, helpdesk_label_pad)

    # multiple input & output dataset
    input_ids = inputs[0]["input_ids"]
    input_mask = inputs[0]["input_mask"]
    input_type_ids = inputs[0]["input_type_ids"]

    if task == "nugget":
        assert len(sentence_ids[0]) == len(sentence_mask[0]) == (len(customer_labels[0]) + len(helpdesk_labels[0]))

        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "input_ids": input_ids, "input_mask": input_mask, "input_type_ids":input_type_ids,
                "sentence_ids": sentence_ids, "sentence_masks": sentence_mask,
                "customer_labels": customer_labels, "helpdesk_labels": helpdesk_labels
            }
        )
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size).batch(32)

    elif task == "quality":
        return []

    else:
        raise ValueError("Task not in (nugget, quality)")

    return dataset, max_turn_number