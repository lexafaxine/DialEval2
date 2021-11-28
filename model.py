import tensorflow as tf
from tensorflow.keras import layers, losses, metrics
import numpy as np
from transformers import BertTokenizer, TFBertModel, BertConfig, XLNetTokenizer, TFXLNetModel, AutoTokenizer, \
    TFAutoModel
from eval_func import normalize
from math import log2
import neural_structured_learning as nsl
from scipy import stats


def check_nan(x, name):
    message = 'checking' + name
    try:
        tf.debugging.check_numerics(x, message=message)
    except Exception as e:
        assert "Checking b : Tensor had NaN values" in e.message


# positional encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rad = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :],
                           d_model)

    # apply sin to even indices in the array; 2i
    angle_rad[:, 0::2] = np.sin(angle_rad[:, 0::2])

    # apply cos to odd indices in the array; 2i + 1
    angle_rad[:, 1::2] = np.cos(angle_rad[:, 1::2])

    pos_encoding = angle_rad[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# point wise feed forward netword
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# encoder transformer layer
class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=rate)
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, attention_mask=mask)  # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout(ffn_output, training=training)

        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


class TransformerEncoder(layers.Layer):
    """A transformer encoder consist of
    1. Input Embedding
    2. Positional Encoding
    3. N encoder layers"""

    def __init__(self, d_model, num_heads, d_ff, maximum_position_encoding, rate=0.1, num_layers=2):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rate=rate)
                           for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


# bert and return the top_vec
class Bert(layers.Layer):

    def __init__(self, language):
        super(Bert, self).__init__()
        # Load Transformers config
        if language == "Chinese":
            bert_name = "bert-base-chinese"
        elif language == "English":
            bert_name = "bert-base-uncased"

        else:
            bert_name = None
            raise ValueError("language must be Chinese or English")

        self.config = BertConfig.from_pretrained(bert_name)
        self.config.output_hidden_states = True
        # Load Bert Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_name, config=self.config)

        # Load the Transformes BERT model
        self.model = TFBertModel.from_pretrained(bert_name, config=self.config)

    def call(self, inputs):
        # inputs = [input_ids, input_mask, input_type_ids, dialogue_length, turn_number, labels]

        input_ids = inputs[0]
        input_mask = inputs[1]
        input_type_ids = inputs[2]
        outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_type_ids,
                             training=True)
        x = outputs["pooler_output"]
        check_nan(x, name="bert_output")
        return x


class XLNet(layers.Layer):

    def __init__(self, language):
        super(XLNet, self).__init__()
        if language == "English":
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            self.model = TFXLNetModel.from_pretrained('xlnet-base-cased')

        elif language == "Chinese":
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")
            self.model = TFAutoModel.from_pretrained("hfl/chinese-xlnet-base")
        else:
            raise ValueError("Language must be English or Chinese!")

    def call(self, inputs):
        input_ids = inputs[0]
        input_mask = inputs[1]
        input_type_ids = inputs[2]
        outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_type_ids)
        x = outputs.last_hidden_state

        x = tf.reduce_mean(x, axis=1)

        check_nan(x, name="xlnet_output")

        return x


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=2000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_sentence_model(plm_name, language, sender, max_len, hidden_size, rnn_dropout=0.1, warmup=1200):
    # an attention-based bilstm model

    if plm_name == "BERT":
        plm = Bert(language=language)

    elif plm_name == "XLNet":
        plm = XLNet(language=language)

    else:
        plm = None
        raise ValueError("Pretrained model name should be specified: XLNet or BERT")

    # define inputs
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")
    dialogue_length = tf.keras.Input(shape=(), dtype=tf.int32, name="dialogue_length")
    turn_number = tf.keras.Input(shape=(), dtype=tf.int32, name="turn_number")

    if sender == "customer":
        # labels = tf.keras.layers.Input(shape=(4,), dtype=tf.float32, name="labels")
        dense_dim = 4
    elif sender == "helpdesk":
        # labels = tf.keras.layers.Input(shape=(3,), dtype=tf.float32, name="labels")
        dense_dim = 3
    else:

        dense_dim = 0
        raise ValueError("sender must be customer or helpdesk")

    inputs = [input_ids, input_mask, input_type_ids, dialogue_length, turn_number]

    # define graph
    plm_output = plm(inputs)  # [Batch, max_len, hidden_size]
    dropout_output = tf.keras.layers.Dropout(0.1)(plm_output)
    dense_output = tf.keras.layers.Dense(dense_dim, activation="relu", name="classification")(dropout_output)

    # keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

    def custom_loss(y_true, y_pred):
        return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred, axis=-1)

    def rnss(y_true, y_pred):
        check_nan(y_pred, name="dense_output")
        pred = tf.nn.softmax(y_pred, axis=-1).numpy()
        truth = y_true.numpy()

        def squared_error(pred, truth):
            return ((pred - truth) ** 2).sum()

        pred, truth = normalize(pred, truth)
        return -log2(np.sqrt(squared_error(pred, truth) / 2))

    def jsd(y_true, y_pred, base=2):
        pred = tf.nn.softmax(y_pred, axis=-1).numpy()
        truth = y_true.numpy()
        m = 1. / 2 * (pred + truth)
        return (stats.entropy(pred, m, base=base)
                + stats.entropy(truth, m, base=base)) / 2.

    # define opt and loss
    # opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, lr=1e-

    # 768: output hidden size of BERT
    learning_rate = CustomSchedule(d_model=768, warmup_steps=warmup)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, lr=1e-5)
    model = tf.keras.Model(inputs=inputs, outputs=dense_output, name="BertND")
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        run_eagerly=True,
        metrics=[rnss]
    )

    return model


def create_dialogue_model(plm_name, max_len, hidden_size, ff_size, heads, dropout, language):
    pass
