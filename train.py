import os
import tensorflow as tf
from pathlib2 import Path
from model import create_dialogue_model
from data_loader import create_dataset, create_processor, create_predict_inputs
from configure import FLAGS
from datetime import datetime
from eval_func import pred_to_submission, evaluate
import pandas as pd

PROJECT_DIR = Path("/content/drive/MyDrive/dialogue")
NUGGET_RESULT = PROJECT_DIR / "dialogue_nugget.csv"
BASELINE_QUALITY_RESULT = PROJECT_DIR / "baseline_quality.csv"
TRANSFORMER_QUALITY_RESULT = PROJECT_DIR / "transformer_quality.csv"

flags = tf.compat.v1.flags


def get_model_name(plm, language, task):
    now = datetime.now()
    date_time = now.strftime("%y%m%d%H%M")
    return plm + "_" + language + "_" + task + "_" + date_time + "/"


class Trainer(object):
    def __init__(self, log_to_tensorboard=True):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        self.logger = tf.get_logger()

        if FLAGS.language not in ["Chinese", "English"]:
            raise ValueError("Language must be Chinese or English!")

        if FLAGS.task not in ["quality", "nugget"]:
            raise ValueError("task must be quality or nugget")

        if FLAGS.encoder not in ["baseline", "transformer"]:
            raise ValueError("encoder must be baseline or transformer")

        self.lang = "zh" if FLAGS.language == "Chinese" else "en"
        self.logger.info("Task: " + str(FLAGS.task))
        self.logger.info("Language: " + str(FLAGS.language))

        # Load Data
        self.train_path, self.dev_path, self.test_path = prepare_data(data_dir=FLAGS.data_dir,
                                                                      language=FLAGS.language)
        self.processor, embedding_size = create_processor(plm=FLAGS.plm, max_len=FLAGS.max_len, language=FLAGS.language)

        self.train_dataset = create_dataset(processor=self.processor, json_path=str(self.train_path),
                                            task=FLAGS.task,
                                            shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                                            batch_size=FLAGS.batch_size)
        self.val_dataset = create_dataset(processor=self.processor, json_path=str(self.dev_path),
                                          task=FLAGS.task,
                                          shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                                          batch_size=FLAGS.batch_size)

        # Create model
        self.model = create_dialogue_model(plm_name=FLAGS.plm, language=FLAGS.language,
                                           max_turn_number=FLAGS.max_turn_number,
                                           embedding_size=embedding_size, max_len=FLAGS.max_len, ff_size=FLAGS.ff_size,
                                           layer_num=FLAGS.layer_num, encoder=FLAGS.encoder,
                                           dropout=FLAGS.dropout, task=FLAGS.task)

        if log_to_tensorboard:
            pass

    def train(self):
        self.logger.info("Start Training Model...")
        model_path = FLAGS.checkpoint_dir + "/" + get_model_name(FLAGS.plm, language=self.lang,
                                                                 task=FLAGS.task)
        os.mkdir(model_path)
        if FLAGS.task == "nugget":
            callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=True,
                                                          verbose=1, monitor="val_rnss",
                                                          save_best_only=True, mode="max")
        else:
            callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=True,
                                                          verbose=1, monitor="val_mean_nmd",
                                                          save_best_only=True, mode="max")
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        train_steps = (4090 // FLAGS.batch_size) + 1
        val_steps = (300 // FLAGS.batch_size) + 1
        history = self.model.fit(x=self.train_dataset, epochs=FLAGS.epoch, verbose="auto", callbacks=callback,
                                 validation_data=self.val_dataset, steps_per_epoch=train_steps,
                                 validation_steps=val_steps)

        return history, model_path

    def validate(self, model_path, mode):

        if mode not in ["test", "validate"]:
            raise ValueError("mode must be test or validate")

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        self.logger.info("Start Validating...")
        self.model.load_weights(model_path)

        if mode == "validate":
            predict_path = self.dev_path
        else:
            predict_path = self.test_path

        predict_inputs = create_predict_inputs(processor=self.processor, json_path=predict_path, task=FLAGS.task)

        if mode == "validate":
            output_file = Path(model_path) / "val_submission.json"
        else:
            output_file = Path(model_path) / "test_submission.json"

        submission = pred_to_submission(inputs=predict_inputs, output_file=output_file, write_to_file=True,
                                        model=self.model, task=FLAGS.task)

        if mode == "validate":
            truth_path = self.dev_path
        else:
            truth_path = str(self.test_path)[:-13] + ".json"
            print(truth_path)

        results = evaluate(task=FLAGS.task, pred_path=output_file, truth_path=truth_path, strict=True)
        # self.logger.info("Evaluate Result: {jsd:" + str(results["jsd"]) + ", rnss:" + str(results["rnss"]) + "}")
        if FLAGS.task == "nugget":
            result_dict = {
                'ckpt': [model_path],
                'language': [FLAGS.language],
                'mode': [mode],
                'jsd': [results["jsd"]],
                'rnss': [results["rnss"]]
            }
            df = pd.DataFrame(result_dict)

            if not os.path.isfile(NUGGET_RESULT):
                df.to_csv(NUGGET_RESULT, index=False)

            else:
                df.to_csv(NUGGET_RESULT, mode='a', header=False, index=False)

        else:
            result_dict = {
                'ckpt': [model_path],
                'language': [FLAGS.language],
                'mode': [mode],
                'rsnod-A': [results["rsnod"]["A"]],
                'rsnod-E': [results["rsnod"]["E"]],
                'rsnod-S': [results["rsnod"]["S"]],
                'nmd-A': [results["nmd"]["A"]],
                'nmd-E': [results["nmd"]["E"]],
                'nmd-S': [results["nmd"]["S"]],
            }
            df = pd.DataFrame(result_dict)

            if FLAGS.encoder == "baseline":
                quality_result = BASELINE_QUALITY_RESULT
            else:
                quality_result = TRANSFORMER_QUALITY_RESULT

            if not os.path.isfile(quality_result):
                df.to_csv(quality_result, index=False)
            else:
                df.to_csv(quality_result, mode='a', header=False, index=False)

        return results


def prepare_data(data_dir, language):
    data_dir = Path(data_dir)

    if language == "Chinese":
        train_path = data_dir / "train_cn.json"
        test_path = data_dir / "test_cn_wo_anno.json"
        dev_path = data_dir / "dev_cn.json"

    else:
        train_path = data_dir / "train_en.json"
        test_path = data_dir / "test_en_wo_anno.json"
        dev_path = data_dir / "dev_en.json"

    if not os.path.isfile(test_path):
        test_path = None

    return train_path, dev_path, test_path


def main(_):
    trainer = Trainer()
    _, model_path = trainer.train()
    trainer.validate(model_path, mode="validate")
    trainer.validate(model_path, mode="test")


if __name__ == '__main__':
    tf.compat.v1.app.run()
