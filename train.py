import os
import tensorflow as tf
from pathlib2 import Path
from model import create_sentence_model, create_dialogue_model
from data_loader import create_dataset, create_inputs, create_predict_input
from configure import FLAGS
from datetime import datetime
from eval_func import pred_to_submission, evaluate
import pandas as pd

PROJECT_DIR = Path("/content/drive/MyDrive/nugget")
NUGGET_RESULT = PROJECT_DIR / "nugget_result.csv"

flags = tf.compat.v1.flags


def get_model_name(plm, sender, language):
    now = datetime.now()
    # BERT_cust_zh_2111161910
    if language == "Chinese":
        lang = "zh"
    else:
        lang = "en"

    date_time = now.strftime("%y%m%d%H%M")
    if sender == "customer" or "helpdesk":
        return plm + "_" + sender[:4] + "_" + lang + date_time
    elif sender == "both":
        return "outputs" + plm + "_" + lang + date_time


class Trainer(object):
    def __init__(self, log_to_tensorboard=True):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        self.logger = tf.get_logger()

        if FLAGS.language not in ["Chinese", "English"]:
            raise ValueError("Language must be Chinese or English!")

        lang = "zh" if FLAGS.language == "Chinese" else "en"
        self.logger.info("Task: " + str(FLAGS.task))
        self.logger.info("Language: " + str(FLAGS.language))

        # Load Data
        self.logger.info("Loading Dataset...")
        self.train_path, self.dev_path, self.test_path = prepare_data(data_dir=FLAGS.data_dir,
                                                                      language=FLAGS.language)

        if FLAGS.embedding == "sentence":
            self.train_cust, self.train_help = get_dataset(mode="train", embedding="sentence", plm=FLAGS.plm,
                                                           language=lang)

            if self.train_cust is None:
                train_data_cust = create_inputs(json_path=self.train_path, plm=FLAGS.plm,
                                                max_len=FLAGS.max_sentence_len,
                                                is_train=True, sender="customer", language=FLAGS.language)
                self.train_cust = create_dataset(inputs=train_data_cust, shuffle_buffer_size=100,
                                                 batch_size=FLAGS.batch_size)
                # BERT_cust_train_zn
                dataset_path = FLAGS.data_dir + "/" + FLAGS.plm + "_" + "cust" + "_train_" + lang
                tf.data.experimental.save(self.train_cust, dataset_path)

            if self.train_help is None:
                train_data_help = create_inputs(json_path=self.train_path, plm=FLAGS.plm,
                                                max_len=FLAGS.max_sentence_len,
                                                is_train=True, sender="helpdesk", language=FLAGS.language)
                self.train_help = create_dataset(inputs=train_data_help, shuffle_buffer_size=100,
                                                 batch_size=FLAGS.batch_size)
                dataset_path = FLAGS.data_dir + "/" + FLAGS.plm + "_" + "help" + "_train_" + lang
                tf.data.experimental.save(self.train_help, dataset_path)

            self.val_cust, self.val_help = get_dataset(mode="val", embedding="sentence", plm=FLAGS.plm,
                                                       language=lang)

            if self.val_cust is None:
                dev_data_c = create_predict_input(json_path=self.dev_path, plm=FLAGS.plm,
                                                  max_len=FLAGS.max_sentence_len,
                                                  sender="customer", language=FLAGS.language)

                self.val_cust = create_dataset(dev_data_c, shuffle_buffer_size=100, batch_size=32)
                dataset_path = FLAGS.data_dir + "/" + FLAGS.plm + "_" + "cust" + "_val_" + lang
                tf.data.experimental.save(self.val_cust, dataset_path)

            if self.val_help is None:
                dev_data_h = create_predict_input(json_path=self.dev_path, plm=FLAGS.plm,
                                                  max_len=FLAGS.max_sentence_len,
                                                  sender="helpdesk", language=FLAGS.language)
                dataset_path = FLAGS.data_dir + "/" + FLAGS.plm + "_" + "help" + "_val_" + lang
                self.val_help = create_dataset(dev_data_h, shuffle_buffer_size=100, batch_size=32)
                tf.data.experimental.save(self.val_help, dataset_path)

            # Cretae model
            self.cust_model = create_sentence_model(plm_name=FLAGS.plm, language=FLAGS.language,
                                                    sender="customer", max_len=FLAGS.max_sentence_len,
                                                    hidden_size=FLAGS.lstm_hidden_size, rnn_dropout=FLAGS.rnn_dropout,
                                                    warmup=FLAGS.warmup)
            self.cust_name = get_model_name(FLAGS.plm, sender="customer", language=FLAGS.language)

            self.help_model = create_sentence_model(plm_name=FLAGS.plm, language=FLAGS.language,
                                                    sender="helpdesk", max_len=FLAGS.max_sentence_len,
                                                    hidden_size=FLAGS.lstm_hidden_size, rnn_dropout=FLAGS.rnn_dropout,
                                                    warmup=FLAGS.warmup)
            self.help_name = get_model_name(FLAGS.plm, sender="helpdesk", language=FLAGS.language)

        if log_to_tensorboard:
            pass

    def train(self, sender):

        if FLAGS.embedding == "sentence":

            if sender == "customer":
                self.logger.info("Start Training Customer Model...")

                ckpt_path = FLAGS.checkpoint_dir + "/" + self.cust_name + "/"

                callback_cust = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                                   save_weights_only=True, verbose=1,
                                                                   monitor="val_rnss", save_best_only=True, mode="max")

                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                history_cust = self.cust_model.fit(x=self.train_cust, epochs=FLAGS.epoch, verbose="auto",
                                                   callbacks=callback_cust, validation_data=self.val_cust)
                tf.keras.utils.plot_model(self.cust_model, ckpt_path + "model.png", show_shapes=True)

                return history_cust, ckpt_path

            elif sender == "helpdesk":

                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
                self.logger.info("Start Training Helpdesk Model...")

                ckpt_path = FLAGS.checkpoint_dir + "/" + self.help_name + "/"
                callback_help = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                                   save_weights_only=True, verbose=1,
                                                                   monitor="val_rnss", save_best_only=True, mode="max")

                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                history_help = self.help_model.fit(x=self.train_help, epochs=FLAGS.epoch, verbose="auto",
                                                   callbacks=callback_help, validation_data=self.val_help)
                tf.keras.utils.plot_model(self.help_model, ckpt_path + "model.png", show_shapes=True)

                return history_help, ckpt_path

        elif FLAGS.embedding == "dialogue":
            pass

    def validate(self, ckpt_cus, ckpt_help):
        if FLAGS.embedding == "sentence":
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
            self.logger.info("Start Evaluating the model...")
            cus_model = create_sentence_model(plm_name=FLAGS.plm, language=FLAGS.language, sender="customer",
                                              max_len=FLAGS.max_sentence_len, hidden_size=FLAGS.lstm_hidden_size,
                                              rnn_dropout=FLAGS.rnn_dropout)
            help_model = create_sentence_model(plm_name=FLAGS.plm, language=FLAGS.language, sender="helpdesk",
                                               max_len=FLAGS.max_sentence_len, hidden_size=FLAGS.lstm_hidden_size,
                                               rnn_dropout=FLAGS.rnn_dropout)

            cus_model.load_weights(ckpt_cus)
            help_model.load_weights(ckpt_help)
            dev_inputs = create_predict_input(json_path=self.dev_path, plm=FLAGS.plm, max_len=FLAGS.max_sentence_len,
                                              sender="both", language=FLAGS.language)

            output_file = Path(ckpt_cus) / "submission.json"

            submission = pred_to_submission(inputs=dev_inputs, cus_model=cus_model, help_model=help_model,
                                            output_file=output_file, write_to_file=True)
            results = evaluate(output_file, self.dev_path, strict=True)

            self.logger.info("Evaluate Result: {jsd:" + str(results["jsd"]) + ", rnss:" + str(results["rnss"]) + "}")

            if FLAGS.task == "nugget":
                result_dict = {
                    'ckpt': [ckpt_cus],
                    'jsd': [results["jsd"]],
                    'rnss': [results["rnss"]]
                }
                df = pd.DataFrame(result_dict)

                if not os.path.isfile(NUGGET_RESULT):
                    df.to_csv(NUGGET_RESULT, index=False)

                else:
                    df.to_csv(NUGGET_RESULT, mode='a', header=False, index=False)

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


def get_dataset(mode, embedding, plm, language):
    if embedding == "sentence":
        cus_dataset_path = FLAGS.data_dir + plm + "_" + "cust_" + mode + "_" + language
        help_dataset_path = FLAGS.data_dir + plm + "_" + "help_" + mode + "_" + language
        if not os.path.isdir(cus_dataset_path):
            cus_dataset = None
        else:
            print("Get dataset from:", cus_dataset_path)
            cus_dataset = tf.data.experimental.load(cus_dataset_path)

        if not os.path.isdir(help_dataset_path):
            help_dataset = None
        else:
            print("Get dataset from:", cus_dataset_path)
            help_dataset = tf.data.experimental.load(help_dataset_path)

        return cus_dataset, help_dataset


def main(_):
    trainer = Trainer()
    if FLAGS.embedding == "sentence":
        _, ckpt_cus = trainer.train(sender="customer")
        _, ckpt_help = trainer.train(sender="helpdesk")
        trainer.validate(ckpt_cus, ckpt_help)


if __name__ == "__main__":
    tf.compat.v1.app.run()
