from pathlib2 import Path
import tensorflow as tf
import numpy as np
from model import create_dialogue_model
from eval_func import pred_to_submission, evaluate
from data_loader import create_predict_inputs, create_processor

PROJECT_DIR = Path(__file__).parent.resolve()


def eval(model_path, data_dir, plm, task, language, output_file):

    processor, embedding_size = create_processor(plm=plm, max_len=512, language=language)

    dialogue_model = create_dialogue_model(plm_name=plm, language=language, max_turn_number=7, task="quality",
                                           embedding_size=embedding_size)
    dialogue_model.load_weights(model_path)
    dev_path = data_dir / "dev_cn.json"

    dev_inputs = create_predict_inputs(processor=processor, json_path=dev_path, task=task)

    output_file = Path(model_path) / "submission.json"

    submission = pred_to_submission(inputs=dev_inputs, output_file=output_file, write_to_file=True,
                                    model=dialogue_model, task="quality")

    results = evaluate(task="quality", pred_path=output_file, truth_path=dev_path, strict=True)

    return results