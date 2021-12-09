from pathlib2 import Path
import tensorflow as tf
import numpy as np
from model import create_dialogue_model
from eval_func import pred_to_submission, evaluate
from data_loader import create_predict_inputs

PROJECT_DIR = Path(__file__).parent.resolve()


def eval(model_path, data_dir, plm, task, max_len, language, output_file):
    dialogue_model = tf.keras.models.load_model(model_path)
    dev_path = data_dir / "dev_cn.json"

    dev_inputs = create_predict_inputs(dev_path, plm=plm, max_len=max_len, language=language, task=task)
    # model_input = (dialogue_id, [input_ids, input_mask, input_type_ids, sentence_ids, sentence_mask,
    # customer_labels, helpdesk_labels])

    submission = pred_to_submission(inputs=dev_inputs, output_file=output_file, write_to_file=True,
                                    model=dialogue_model, task="nugget")
