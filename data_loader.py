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
        "sentence_masks": None,
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


def create_inputs(json_path, plm, max_len, is_train, language, task):
    if is_train:
        mode = "train"
        processor = Processor(plm=plm, max_len=max_len, language=language)
        raw_data = json.load(open(json_path))

        for dialogue in raw_data:
            example = processor.process_dialogue(dialogue, mode="train", task="nugget")
            data = transfer_example_to_input(example, is_test=False)
            input_ids = data[0]
            input_mask = data[1]
            input_type_ids = data[2]
            sentence_ids = data[3]
            sentence_masks = data[4]
            customer_labels = data[5]
            helpdesk_labels = data[6]
            quality_labels = data[7]

            if task == "nugget":
                yield {
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "input_type_ids": input_type_ids,
                    "sentence_ids": sentence_ids,
                    "sentence_masks": sentence_masks,
                    "customer_labels": customer_labels,
                    "helpdesk_labels": helpdesk_labels
                }


def create_dataset(data, task, shuffle_buffer_size=200, batch_size=32):

    if task == "nugget":
        dataset = tf.data.Dataset.from_generator(
            lambda: (x for x in data),
            output_types=({"input_ids": tf.int32,
                           "input_mask": tf.int32,
                           "input_type_ids": tf.int32,
                           "sentence_ids": tf.int32,
                           "sentence_masks": tf.int32,
                           "customer_labels": tf.float32,
                           "helpdesk_labels": tf.float32})
        )

        dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size=batch_size)

        return dataset

    elif task == "quality":
        return []

    else:
        raise ValueError("Task not in (nugget, quality)")
