import argparse
import sys
from pathlib2 import Path
import tensorflow as tf

PROJECT_DIR = Path("/content/drive/MyDrive/dialogue")


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    project_dir = str(PROJECT_DIR)

    # Data loading params
    parser.add_argument("--data_dir", default="./dataset", type=str,
                        help="Path of train, dev and test")
    parser.add_argument("--checkpoint_dir", default=project_dir, type=str,
                        help="Directory where the checkpoints are saved")
    parser.add_argument("--shuffle_buffer_size", default=100, type=int,
                        help="Dataset shuffle buffer size")

    # Task detail
    parser.add_argument("--mode", default=None, type=str,
                        help="Train, evaluate or predict", required=True)
    parser.add_argument("--language", default=None, type=str,
                        help="English or Chinese", required=True)
    parser.add_argument("--task", default=None, type=str,
                        help="Nugget or Quality(both lower case)", required=True)

    # Preprocessing & Model
    parser.add_argument("--max_dialogue_length", default=512, type=int,
                        help="Max diague length for dialogue-level model")
    parser.add_argument("--max_sentence_len", default=100, type=int,
                        help="Max sentence length for sentence-level model")
    parser.add_argument("--plm", default=None, type=str,
                        help="Pretrained Language Model name: BERT or XLNet", required=True)

    # Model Hyper-parameters
    parser.add_argument("--lstm_hidden_size", default=100, type=int,
                        help="Dimensionality of RNN hidden (default: 100)")
    parser.add_argument("--transformer_hidden_size", default=768, type=int,
                        help="Dimensionality of Transformer Encode Block(default:768)")
    parser.add_argument("--ff_size", default=512, type=int,
                        help="ff size of attention in transformer")
    parser.add_argument("--heads", default=8, type=int,
                        help="Number of head of the multi head attention")
    parser.add_argument("--rnn-dropout", default=0.3, type=int,
                        help="Dropout rate of RNN")
    parser.add_argument("--dropout", default=0.1, type=int,
                        help="Dropout rate of the inputs of Dense layer")

    # Training
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch Size")
    parser.add_argument("--epoch", default=20, type=int,
                        help="Number of Epoch")
    parser.add_argument("--embedding", default=None, type=str,
                        help="Embedding level: sentence or dialogue", required=True)
    parser.add_argument("--warmup", default=1200, type=int,
                        help="warm up steps of optimizer")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print("")

    return args


FLAGS = parse_args()
