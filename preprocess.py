import json
import os
from transformers import BertTokenizer, TFBertModel, BertConfig, AutoTokenizer, XLNetTokenizer
from collections import Counter
import numpy as np
from itertools import chain

# split custom labels and helpdesk labels to train separately in sentence level

QUALITY_MEASURES = ("A", "E", "S")
QUALITY_SCALES = ('2', '1', '0', '-1', '-2')
CUSTOMER_NUGGET_TYPES = ('CNUG0', 'CNUG', 'CNUG*', 'CNaN')
HELPDESK_NUGGET_TYPES = ('HNUG', 'HNUG*', 'HNaN')
MAX_TURN_NUMBER = 7
C_NUGGET_TYPES_TO_INDEX = {
    "PAD": 0,
    "CNaN": 1,
    "CNUG": 2,
    "CNUG*": 3,
    "CNUG0": 4,

}

H_NUGGET_TYPES_TO_INDEX = {
    "PAD": 0,
    "HNaN": 1,
    "HNUG": 2,
    "HNUG*": 3,
}


class DialogueData(object):
    def __init__(self, dialogue_id, input_ids, input_mask, input_type_ids, sentence_ids, sentence_mask,
                 customer_labels,
                 helpdesk_labels, quality_labels):
        self.dialogue_id = dialogue_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.sentence_ids = sentence_ids
        self.sentence_masks = sentence_mask
        self.customer_labels = customer_labels
        self.helpdesk_labels = helpdesk_labels
        self.quality_labels = quality_labels


def parse_nugget(annotations, dialogue_length):
    customer_labels = []
    helpdesk_labels = []
    labels = []
    # utterance order is [customer, helpdesk, customer, helpdesk, ...]

    for i in range(0, dialogue_length, 2):
        # deal with customer annotations
        nugget_types = CUSTOMER_NUGGET_TYPES

        c_nuggets = [h[i] for h in [anno["nugget"] for anno in annotations]]
        count = Counter(c_nuggets)
        distribution = []
        for nugget_type in nugget_types:
            distribution.append(count.get(nugget_type, 0))

        distribution = np.array(distribution, dtype=np.float32)
        assert distribution.sum() == sum(count.values()), distribution
        distribution = distribution / distribution.sum()

        customer_labels.append(distribution)

    for i in range(1, dialogue_length, 2):
        # deal with helpdesk annotations
        nugget_types = HELPDESK_NUGGET_TYPES

        h_nuggets = [h[i] for h in [anno["nugget"] for anno in annotations]]
        count = Counter(h_nuggets)
        distribution = []
        for nugget_type in nugget_types:
            # order
            distribution.append(count.get(nugget_type, 0))

        distribution = np.array(distribution, dtype=np.float32)
        assert distribution.sum() == sum(count.values()), distribution
        distribution = distribution / distribution.sum()

        helpdesk_labels.append(distribution)

    return customer_labels, helpdesk_labels


def parse_quality(annotations):
    label = {}

    for anno in annotations:
        for measure in QUALITY_MEASURES:
            label.setdefault(measure, [])
            label[measure].append(anno["quality"][measure])

    for measure, values in label.items():
        distribution = []
        count = Counter(values)
        for scale in QUALITY_SCALES:
            distribution.append(count.get(int(scale), 0))

        distribution = np.array(distribution, dtype=np.float32)
        assert distribution.sum() == sum(count.values())
        distribution /= distribution.sum()

        label[measure] = distribution

    label = [label[measure] for measure in QUALITY_MEASURES]

    return np.stack(label)


def truncator(tokenized_dialogue, sep_token, max_len):
    turn_number = len(tokenized_dialogue)
    max_utterance_len = max_len // turn_number

    length_now = 0

    truncated_dialogue = []

    for utterance in tokenized_dialogue:
        utterance = utterance[:max_utterance_len - 1]
        utterance.append(sep_token)
        truncated_dialogue.append(utterance)
        length_now += len(utterance)
    assert length_now <= 512

    return truncated_dialogue


class Processor(object):
    def __init__(self, plm, max_len, language):
        self.max_len = max_len

        self.plm = plm

        if self.plm == "BERT":
            if language == "Chinese":
                plm_name = "bert-base-chinese"
            else:
                plm_name = "bert-base-cased"
            config = BertConfig.from_pretrained(plm_name)
            config.output_hidden_states = False

            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=plm_name, config=config)
            self.pad_vid = self.tokenizer.convert_tokens_to_ids('[PAD]')
            self.sep_vid = self.tokenizer.convert_tokens_to_ids('[SEP]')
            self.cls_vid = self.tokenizer.convert_tokens_to_ids('[CLS]')
        elif self.plm == "XLNet":
            if language == "Chinese":
                self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")
            else:
                self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            self.pad_vid = self.tokenizer.convert_tokens_to_ids('<pad>')
            self.sep_vid = self.tokenizer.convert_tokens_to_ids('<sep>')
            self.cls_vid = self.tokenizer.convert_tokens_to_ids('<cls>')

        else:
            raise ValueError("xxxxxxx")

        self.len_tokens = ["[len1]", "[len2]", "[len3]", "[len4]", "[len5]", "[len6]", "[len7]", "[len8]"]
        self.trn_tokens = ["[trn1]", "[trn2]", "[trn3]", "[trn4]", "[trn5]", "[trn6]"]
        self.sdr_tokens = ["[customer]", "[helpdesk]"]

        self.tokenizer.add_tokens(self.len_tokens, special_tokens=True)
        self.tokenizer.add_tokens(self.trn_tokens, special_tokens=True)
        self.tokenizer.add_tokens(self.sdr_tokens, special_tokens=True)

        self.tokenizer_length = len(self.tokenizer)

    def process_dialogue(self, dialogue, is_test):

        turn_number = len(dialogue["turns"])
        if is_test is False:
            annotations = dialogue["annotations"]
            customer_labels, helpdesk_labels = parse_nugget(annotations=annotations, dialogue_length=turn_number)
            quality_labels = parse_quality(annotations)

        else:
            quality_labels = [[1/len(QUALITY_SCALES)] * len(QUALITY_SCALES)] * len(QUALITY_MEASURES)
            customer_turn = (turn_number // 2) + 1 if turn_number % 2 == 1 else turn_number // 2
            helpdesk_turn = turn_number // 2

            customer_labels = [[1. / len(CUSTOMER_NUGGET_TYPES)] * len(CUSTOMER_NUGGET_TYPES)] * customer_turn
            helpdesk_labels = [[1. / len(HELPDESK_NUGGET_TYPES)] * len(HELPDESK_NUGGET_TYPES)] * helpdesk_turn

        # assert turn_number == len(quality_labels)
        assert turn_number == len(customer_labels) + len(helpdesk_labels)

        turns = dialogue["turns"]
        dialogue_id = dialogue["id"]

        dialogue_length = 0
        tokenized_dialogue = []

        for i in range(turn_number):
            utterance = " ".join(turns[i]["utterances"])
            utterance_token = self.tokenizer.tokenize(utterance)
            # cls,  sep,  len,  trn
            len_token = self.len_tokens[turn_number - 1]
            trn_token = self.trn_tokens[i // 2]
            sdr_token = self.sdr_tokens[0] if i % 2 == 0 else self.sdr_tokens[1]

            utterance_token = [len_token] + [trn_token] + [sdr_token] + utterance_token

            if self.plm == "BERT":
                utterance_token = ['[CLS]'] + utterance_token + ['[SEP]']
            else:
                utterance_token = utterance_token + ['<sep>'] + ['<cls>']

            dialogue_length += len(utterance_token)
            tokenized_dialogue.append(utterance_token)

        if dialogue_length > self.max_len:
            # max length of bert and xlnet
            sep_token = "[SEP]" if self.plm == "BERT" else "<sep>"
            tokenized_dialogue = truncator(tokenized_dialogue, sep_token, self.max_len)

        tokenized_dialogue = list(chain.from_iterable(tokenized_dialogue))

        dialogue_idxs = self.tokenizer.convert_tokens_to_ids(tokenized_dialogue)

        _segs = [-1] + [i for i, t in enumerate(dialogue_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

        segments_ids = []

        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        if self.plm == "BERT":
            sentence_ids = [i for i, t in enumerate(dialogue_idxs) if t == self.cls_vid]
        elif self.plm == "XLNet":
            sentence_ids = [i for i, t in enumerate(dialogue_idxs) if t == self.sep_vid]
        else:
            raise ValueError("plm")
        assert len(sentence_ids) == (len(customer_labels) + len(helpdesk_labels))

        # pad input

        padding_length = self.max_len - len(dialogue_idxs)
        input_mask = [0] * self.max_len

        if segments_ids == []:
            print(dialogue_id, tokenized_dialogue)

        if padding_length > 0:
            dialogue_idxs = dialogue_idxs + ([self.pad_vid] * padding_length)
            if segments_ids[-1] == 1:
                segments_ids = segments_ids + ([0] * padding_length)
            else:
                segments_ids = segments_ids + ([1] * padding_length)
        for j in range(self.max_len):
            if dialogue_idxs[j] != self.pad_vid:
                input_mask[j] = 1

        # pad sentence ids
        sentence_padding_length = MAX_TURN_NUMBER - turn_number
        for i in range(sentence_padding_length):
            sentence_ids.append(-1)

        sentence_mask = [0] * MAX_TURN_NUMBER
        for i in range(MAX_TURN_NUMBER):
            if sentence_ids[i] != -1:
                sentence_mask[i] = 1
        for j in range(MAX_TURN_NUMBER):
            if sentence_mask[j] == 0:
                sentence_ids[j] = 0

        # pad label
        customer_max_turn = (MAX_TURN_NUMBER // 2) + 1 if MAX_TURN_NUMBER % 2 == 1 else MAX_TURN_NUMBER // 2
        customer_padding_length = customer_max_turn - len(customer_labels)

        for i in range(customer_padding_length):
            customer_labels.append([1. / len(CUSTOMER_NUGGET_TYPES)] * len(CUSTOMER_NUGGET_TYPES))

        helpdesk_max_turn = MAX_TURN_NUMBER // 2
        helpdesk_padding_length = helpdesk_max_turn - len(helpdesk_labels)

        for i in range(helpdesk_padding_length):
            helpdesk_labels.append([1. / len(HELPDESK_NUGGET_TYPES)] * len(HELPDESK_NUGGET_TYPES))

        # print(len(sentence_ids), len(customer_labels), len(helpdesk_labels), len(sentence_mask))

        assert len(sentence_ids) == (len(customer_labels) + len(helpdesk_labels)) == len(sentence_mask)

        return DialogueData(dialogue_id=dialogue_id, input_ids=dialogue_idxs, input_mask=input_mask,
                            input_type_ids=segments_ids, sentence_ids=sentence_ids,
                            customer_labels=customer_labels, helpdesk_labels=helpdesk_labels,
                            quality_labels=quality_labels, sentence_mask=sentence_mask)
