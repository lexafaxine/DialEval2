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
NUGGET_RESULT = PROJECT_DIR / "nugget_result.csv"
QUALITY_RESULT = PROJECT_DIR / "quality.csv"

flags = tf.compat.v1.flags


def get_model_name(plm, language, task):
    now = datetime.now()
    date_time = now.strftime("%y%m%d%H%M")
    return plm + "_" + language + "_" + task + "_" + date_time


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
            callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False,
                                                          verbose=1, monitor="val_rnss",
                                                          save_best_only=True, mode="max")
        else:
            callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False,
                                                          verbose=1, monitor="val_nmd_A",
                                                          save_best_only=True, mode="max")
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        train_steps = (4090 // FLAGS.batch_size) + 1
        val_steps = (300 // FLAGS.batch_size) + 1
        history = self.model.fit(x=self.train_dataset, epochs=FLAGS.epoch, verbose="auto", callbacks=callback,
                                 validation_data=self.val_dataset, steps_per_epoch=train_steps,
                                 validation_steps=val_steps)

        return history, model_path

    def validate(self, model_path):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        self.logger.info("Start Validating...")
        dialogue_model = tf.keras.models.load_model(model_path)
        dev_inputs = create_predict_inputs(processor=self.processor, json_path=self.dev_path, task=FLAGS.task)

        output_file = Path(model_path) / "submission.json"

        submission = pred_to_submission(inputs=dev_inputs, output_file=output_file, write_to_file=True,
                                        model=dialogue_model, task=FLAGS.task)

        results = evaluate(task=FLAGS.task, pred_path=output_file, truth_path=self.dev_path, strict=True)
        # self.logger.info("Evaluate Result: {jsd:" + str(results["jsd"]) + ", rnss:" + str(results["rnss"]) + "}")
        if FLAGS.task == "nugget":
            result_dict = {
                'ckpt': [model_path],
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
                'jsd': [results["jsd"]],
                'rnss': [results["rnss"]]
            }
            df = pd.DataFrame(result_dict)

            if not os.path.isfile(QUALITY_RESULT):
                df.to_csv(QUALITY_RESULT, index=False)
            else:
                df.to_csv(QUALITY_RESULT, mode='a', header=False, index=False)

        return results


def prepare_data(data_dir, language):
    data_dir = Path(data_dir)

    if language == "Chinese":
        train_path = data_dir / "train_cn.json"
        test_path = data_dir / "test_cn.json"
        dev_path = data_dir / "dev_cn.json"

    else:
        train_path = data_dir / "train_en.json"
        test_path = data_dir / "test_en.json"
        dev_path = data_dir / "dev_en.json"

    if not os.path.isfile(test_path):
        test_path = None

    return train_path, dev_path, test_path


def main(_):
    trainer = Trainer()
    _, model_path = trainer.train()
    trainer.validate(model_path)


if __name__ == '__main__':
    tf.compat.v1.app.run()
