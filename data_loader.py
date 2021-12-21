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


def create_predict_inputs(processor, json_path, task):
    raw_data = json.load(open(json_path))

    inputs = []

    # use numpy arrays
    for dialogue in raw_data:
        example = processor.process_dialogue(dialogue, is_test=True)
        data = transfer_example_to_input(example, is_test=True)
        dialogue_id = data[0]
        input_ids = tf.expand_dims(tf.convert_to_tensor(data[1]), axis=0)
        input_mask = tf.expand_dims(tf.convert_to_tensor(data[2]), axis=0)
        input_type_ids = tf.expand_dims(tf.convert_to_tensor(data[3]), axis=0)
        sentence_ids = tf.expand_dims(tf.convert_to_tensor(data[4]), axis=0)
        sentence_mask = tf.expand_dims(tf.convert_to_tensor(data[5]), axis=0)
        customer_labels = tf.expand_dims(tf.convert_to_tensor(data[6]), axis=0)
        helpdesk_labels = tf.expand_dims(tf.convert_to_tensor(data[7]), axis=0)
        quality_labels = tf.expand_dims(tf.convert_to_tensor(data[8]), axis=0)

        if task == "nugget":
            model_input = (dialogue_id, [input_ids, input_mask, input_type_ids, sentence_ids, sentence_mask,
                                         customer_labels, helpdesk_labels])
        elif task == "quality":
            model_input = (dialogue_id, [input_ids, input_mask, input_type_ids, sentence_ids, sentence_mask,
                                         quality_labels])
        else:
            model_input = None
            raise ValueError

        inputs.append(model_input)

    return inputs


def create_processor(plm, max_len, language):
    processor = Processor(plm=plm, max_len=max_len, language=language)

    return processor, len(processor.tokenizer)


def create_inputs(processor, json_path, task):
    raw_data = json.load(open(json_path))

    for dialogue in raw_data:
        example = processor.process_dialogue(dialogue, is_test=False)
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

        elif task == "quality":
            yield {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "input_type_ids": input_type_ids,
                "sentence_ids": sentence_ids,
                "sentence_masks": sentence_masks,
                "quality_labels": quality_labels
            }
        else:
            raise ValueError


def create_dataset(processor, json_path, task, shuffle_buffer_size=200, batch_size=4):
    from functools import partial
    if task == "nugget":
        dataset = tf.data.Dataset.from_generator(
            partial(create_inputs, processor, json_path, task),
            output_types=({"input_ids": tf.int32,
                           "input_mask": tf.int32,
                           "input_type_ids": tf.int32,
                           "sentence_ids": tf.int32,
                           "sentence_masks": tf.int32,
                           "customer_labels": tf.float32,
                           "helpdesk_labels": tf.float32}),
        )

        dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size=batch_size).repeat()

        return dataset

    elif task == "quality":
        dataset = tf.data.Dataset.from_generator(
            partial(create_inputs, processor, json_path, task),
            output_types=({"input_ids": tf.int32,
                           "input_mask": tf.int32,
                           "input_type_ids": tf.int32,
                           "sentence_ids": tf.int32,
                           "sentence_masks": tf.int32,
                           "quality_labels": tf.float32})
        )

        dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size=batch_size).repeat()

        return dataset

    else:
        raise ValueError("Task not in (nugget, quality)")
