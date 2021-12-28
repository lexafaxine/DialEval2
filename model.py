import tensorflow as tf
from tensorflow.keras import layers, losses, metrics
import numpy as np
from transformers import BertTokenizer, TFBertModel, BertConfig, XLNetTokenizer, TFXLNetModel, AutoTokenizer, \
    TFAutoModel
from eval_func import normalize
from math import log2
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

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        mask = tf.expand_dims(mask, axis=1)
        attn_output = self.mha(x, x, x, attention_mask=mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)

        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


class TransformerEncoder(layers.Layer):
    """A transformer encoder consist of
    1. Input Embedding
    2. Positional Encoding
    3. N encoder layers"""

    def __init__(self, d_model, num_heads, d_ff, maximum_position_encoding=10000, rate=0.1, num_layers=2):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rate=rate)
                           for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        attn_mask = tf.cast(tf.math.equal(mask, 0), tf.float32)
        # sentence_masks = mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
        # tf.print(sentence_masks.shape)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x=x, training=training, mask=attn_mask)  # (batch_size, input_seq_len,
            # d_model)

        return x


class NuggetDense(layers.Layer):
    """Get custom and helpdesk label's indices
       A multi output model"""

    def __init__(self, customer_dim, helpdesk_dim, max_turn_number, name=None):
        super(NuggetDense, self).__init__()
        self.customer_dense = layers.Dense(customer_dim, activation='relu')
        self.helpdesk_dense = layers.Dense(helpdesk_dim, activation='relu')
        # assume order is [customer, helpdesk, customer, ...]
        self.customer_indices = tf.range(start=0, delta=2, limit=max_turn_number)
        self.helpdesk_indices = tf.range(start=1, delta=2, limit=max_turn_number)

        self.max_turn_number = max_turn_number

    def call(self, x, mask):
        customer_output = tf.gather(x, indices=self.customer_indices, axis=1)
        helpdesk_output = tf.gather(x, indices=self.helpdesk_indices, axis=1)
        assert_op = tf.debugging.assert_equal(tf.shape(customer_output)[1] + tf.shape(helpdesk_output)[1],
                                              self.max_turn_number)
        customer_mask = tf.cast(tf.gather(mask, axis=1, indices=self.customer_indices), dtype=tf.float32)
        helpdesk_mask = tf.cast(tf.gather(mask, axis=1, indices=self.helpdesk_indices), dtype=tf.float32)

        customer_logits = self.customer_dense(customer_output)
        helpdesk_logits = self.helpdesk_dense(helpdesk_output)

        return customer_logits, helpdesk_logits


class QualityDense(layers.Layer):
    '''input = [Batch_size, hidden_size] if encoder=baseline, or
    input = [Batch_size, max_turn_number, hidden_size] (need a pooler)'''

    def __init__(self, encoder):
        super(QualityDense, self).__init__()
        self.pooler = layers.GlobalAveragePooling1D()
        self.encoder = encoder
        self.dense_layers = [layers.Dense(5, activation='relu') for _ in range(3)]

    def call(self, x):
        quality_logits = []
        if self.encoder == "baseline":
            # input = [Batch_size, hidden_size=768]
            dialogue_repr = x

        else:
            # encoder = "transformer" and input = [Batch_size, max_turn_number, hidden_size]
            dialogue_repr = self.pooler(x)

        for i in range(3):
            quality_label = self.dense_layers[i](dialogue_repr)
            quality_logits.append(quality_label)

        return tf.stack(quality_logits, axis=1)


class NuggetSoftmax(layers.Layer):

    def __init__(self, max_turn_number, name=None):
        super(NuggetSoftmax, self).__init__()
        self.customer_indices = tf.range(start=0, delta=2, limit=max_turn_number)
        self.helpdesk_indices = tf.range(start=1, delta=2, limit=max_turn_number)

        self.max_turn_number = max_turn_number

    def call(self, inputs, mask):
        customer_logits = inputs["customer_logits"]
        helpdesk_logits = inputs["helpdesk_logits"]
        customer_labels = inputs["customer_labels"]
        helpdesk_labels = inputs["helpdesk_labels"]

        customer_loss = tf.nn.softmax_cross_entropy_with_logits(customer_labels, customer_logits, axis=-1)
        helpdesk_loss = tf.nn.softmax_cross_entropy_with_logits(helpdesk_labels, helpdesk_logits, axis=-1)
        customer_mask = tf.cast(tf.gather(mask, axis=1, indices=self.customer_indices), dtype=tf.float32)
        helpdesk_mask = tf.cast(tf.gather(mask, axis=1, indices=self.helpdesk_indices), dtype=tf.float32)

        customer_loss = tf.reduce_sum(customer_loss * customer_mask, axis=-1)
        helpdesk_loss = tf.reduce_sum(helpdesk_loss * helpdesk_mask, axis=-1)
        self.add_loss(tf.reduce_mean(customer_loss) + tf.reduce_mean(helpdesk_loss))
        customer_probs = tf.nn.softmax(customer_logits, axis=-1) * customer_mask[:, :, None]
        helpdesk_probs = tf.nn.softmax(helpdesk_logits, axis=-1) * helpdesk_mask[:, :, None]
        # validate
        cust_squared_error = tf.reduce_sum(tf.math.squared_difference(customer_probs, customer_labels), axis=-1)
        help_squared_error = tf.reduce_sum(tf.math.squared_difference(helpdesk_probs, helpdesk_labels), axis=-1)

        customer_rnss = -tf.experimental.numpy.log2(tf.math.sqrt(cust_squared_error / 2)) * customer_mask
        helpdesk_rnss = -tf.experimental.numpy.log2(tf.math.sqrt(help_squared_error / 2)) * helpdesk_mask

        customer_turn = tf.math.count_nonzero(customer_mask, axis=-1)
        helpdesk_turn = tf.math.count_nonzero(helpdesk_mask, axis=-1)
        customer_rnss = tf.math.divide(tf.reduce_sum(customer_rnss, axis=-1), tf.cast(customer_turn, dtype=tf.float32))
        helpdesk_rnss = tf.math.divide(tf.reduce_sum(helpdesk_rnss, axis=-1), tf.cast(helpdesk_turn, dtype=tf.float32))

        rnss = (tf.reduce_mean(customer_rnss) + tf.reduce_mean(helpdesk_rnss)) / 2
        self.add_metric(rnss, name="rnss")

        return customer_probs, helpdesk_probs


class QualitySoftmax(layers.Layer):

    def __init__(self):
        super(QualitySoftmax, self).__init__()

    def call(self, inputs):
        # inputs = [Batch, [3,5]]
        quality_logits = inputs["quality_logits"]
        quality_labels = inputs["quality_labels"]

        loss = tf.nn.softmax_cross_entropy_with_logits(quality_labels, quality_logits, axis=-1)  # [batch, 3]
        mean_loss = tf.reduce_mean(loss, axis=-1)

        self.add_loss(tf.reduce_mean(mean_loss))
        # softmax
        quality_probs = []

        for logits in quality_logits:
            quality_probs.append(tf.nn.softmax(logits, axis=-1))

        # validate
        def get_score(inputs):
            accuracy = []
            logits = inputs[0]
            probs = tf.nn.softmax(logits, axis=-1)
            labels = inputs[1]
            for i in range(3):
                cum_p, cum_q = tf.math.cumsum(probs[i]), tf.math.cumsum(labels[i])
                accuracy.append(tf.math.reduce_sum(tf.math.abs(cum_p - cum_q)) / len(cum_p))

            return tf.convert_to_tensor(accuracy)

        score_inputs = tf.stack([quality_logits, quality_labels], axis=1)

        batch_accuracy = tf.map_fn(get_score, score_inputs,
                                   fn_output_signature=tf.TensorSpec(shape=(3), dtype=tf.float32))

        scores = -tf.experimental.numpy.log2(tf.reduce_mean(batch_accuracy, axis=0))

        self.add_metric(scores[0], name="nmd_A")
        self.add_metric(scores[1], name="nmd_E")
        self.add_metric(scores[2], name="nmd_S")

        return quality_probs


# bert and return the top_vec
class Bert(layers.Layer):

    def __init__(self, language, embedding_size, encoder):
        super(Bert, self).__init__()
        # Load Transformers config
        if language == "Chinese":
            bert_name = "bert-base-chinese"
        elif language == "English":
            bert_name = "bert-base-uncased"

        else:
            bert_name = None
            raise ValueError("language must be Chinese or English")
        self.encoder = encoder
        self.config = BertConfig.from_pretrained(bert_name)
        self.config.output_hidden_states = True
        # Load Bert Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_name, config=self.config)

        # Load the Transformes BERT model
        self.model = TFBertModel.from_pretrained(bert_name, config=self.config)
        self.model.resize_token_embeddings(embedding_size)

    def call(self, inputs):
        # inputs = [input_ids, input_mask, input_type_ids, sentence_ids, sentence_mask]

        input_ids = inputs["input_ids"]
        input_mask = inputs["input_mask"]
        input_type_ids = inputs["input_type_ids"]
        outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_type_ids,
                             training=True)
        if self.encoder == "transformer":
            x = outputs["last_hidden_state"]
            x *= tf.math.sqrt(tf.cast(768, tf.float32))
            # check_nan(x, name="bert_output")

            sentence_ids = inputs["sentence_ids"]
            sentence_masks = inputs["sentence_masks"]

            sents_vec = tf.gather(x, indices=sentence_ids, batch_dims=1)
            sents_vec = sents_vec * tf.cast(sentence_masks[:, :, None], dtype=tf.float32)

            # check_nan(sents_vec, name="sentence vec")

            return sents_vec  # [Batch_size, max_turn_number, hidden_size=768]

        elif self.encoder == "baseline":
            return outputs["pooler_output"]  # [Batch_size, hidden_size=768]
        else:
            raise ValueError("encoder neither transformer nor baseline")


class XLNet(layers.Layer):

    def __init__(self, language, embedding_size, encoder):
        super(XLNet, self).__init__()
        if language == "English":
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            self.model = TFXLNetModel.from_pretrained('xlnet-base-cased')

        elif language == "Chinese":
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")
            self.model = TFAutoModel.from_pretrained("hfl/chinese-xlnet-base")
        else:
            raise ValueError("Language must be English or Chinese!")

        self.model.resize_token_embeddings(embedding_size)
        self.pooler = layers.GlobalAveragePooling1D()
        self.encoder = encoder

    def call(self, inputs):
        # inputs = [input_ids, input_mask, input_type_ids, sentence_ids, sentence_mask]
        input_ids = inputs["input_ids"]
        input_mask = inputs["input_mask"]
        input_type_ids = inputs["input_type_ids"]
        outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_type_ids)

        if self.encoder == "transformer":
            x = outputs.last_hidden_state
            x *= tf.math.sqrt(tf.cast(768, tf.float32))

            check_nan(x, name="xlnet_output")
            sentence_ids = inputs["sentence_ids"]
            sentence_masks = inputs["sentence_masks"]

            sents_vec = tf.gather(x, indices=sentence_ids, batch_dims=1)
            sents_vec = sents_vec * tf.cast(sentence_masks, dtype=tf.float32)

            check_nan(sents_vec, name="sentence vec")

            return sents_vec  # [Batch_size, max_turn_number, hidden_size=768]

        elif self.encoder == "baseline":
            last_hidden_states = outputs.last_hidden_state

            return self.pooler(last_hidden_states)  # [Batch_size, hidden_size=768]

        else:
            raise ValueError("encoder neither transformer nor baseline")


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


def create_dialogue_model(plm_name, language, max_turn_number, task, embedding_size,
                          encoder="transformer", max_len=512, hidden_size=768, ff_size=200, heads=8, layer_num=1,
                          dropout=0.1):
    # tf.keras.backend.set_floatx('float16')
    if task not in ["nugget", "quality"]:
        raise ValueError("task must be nugget or quality")

    if encoder not in ["transformer", "baseline"]:
        raise ValueError("encoder must be transformer or baseline")

    if plm_name == "BERT":
        plm = Bert(language=language, embedding_size=embedding_size, encoder=encoder)

    elif plm_name == "XLNet":
        plm = XLNet(language=language, embedding_size=embedding_size, encoder=encoder)

    else:
        raise ValueError("plm not in (BERT, XLNet)")

    customer_turn = (max_turn_number // 2) + 1 if max_turn_number % 2 == 1 else (max_turn_number // 2)
    helpdesk_turn = (max_turn_number // 2)

    # define inputs
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")
    sentence_ids = tf.keras.Input(shape=(max_turn_number,), dtype=tf.int32, name="sentence_ids")
    sentence_masks = tf.keras.Input(shape=(max_turn_number,), dtype=tf.int32, name="sentence_masks")
    customer_labels = tf.keras.Input(shape=(customer_turn, 4), dtype=tf.float32, name="customer_labels")
    helpdesk_labels = tf.keras.Input(shape=(helpdesk_turn, 3), dtype=tf.float32, name="helpdesk_labels")
    quality_labels = tf.keras.Input(shape=(3, 5), dtype=tf.float32, name="quality_labels")

    # define graph1
    if task == "nugget":
        encoder = TransformerEncoder(d_model=hidden_size, num_heads=heads, d_ff=ff_size, rate=dropout,
                                     num_layers=layer_num)

        nugget_dense = NuggetDense(customer_dim=4, helpdesk_dim=3, max_turn_number=max_turn_number,
                                   name="customer dense")
        nugget_softmax = NuggetSoftmax(max_turn_number=max_turn_number, name="custom Softmax")

        inputs = [input_ids, input_mask, input_type_ids, sentence_ids, sentence_masks, customer_labels, helpdesk_labels]

        plm_inputs = {
            "input_ids": input_ids, "input_mask": input_mask, "input_type_ids": input_type_ids,
            "sentence_ids": sentence_ids, "sentence_masks": sentence_masks
        }
        sents_vec = plm(inputs=plm_inputs)  # [Batch, max_turn_number, hidden_size]
        sents_vec = encoder(x=sents_vec, training=True, mask=sentence_masks)

        customer_logits, helpdesk_logits = nugget_dense(sents_vec, mask=sentence_masks)

        softmax_inputs = {
            "customer_logits": customer_logits, "helpdesk_logits": helpdesk_logits,
            "customer_labels": customer_labels, "helpdesk_labels": helpdesk_labels
        }
        customer_probs, helpdesk_probs = nugget_softmax(softmax_inputs, mask=sentence_masks)

        outputs = [customer_probs, helpdesk_probs]

        opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, lr=2e-5)

        # warm up opt
        learning_rate = CustomSchedule(d_model=768, warmup_steps=4000)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dialogue_nugget")
        model.compile(optimizer=opt, run_eagerly=True)

        return model

    else:
        # task == quality
        inputs = [input_ids, input_mask, input_type_ids, sentence_ids, sentence_masks, quality_labels]

        plm_inputs = {
            "input_ids": input_ids, "input_mask": input_mask, "input_type_ids": input_type_ids,
            "sentence_ids": sentence_ids, "sentence_masks": sentence_masks
        }

        if encoder == "baseline":
            dialogue_repr = plm(plm_inputs)
            quality_dense = QualityDense(encoder=encoder)
            quality_logits = quality_dense(dialogue_repr)

        else:
            # encoder == "transformer"
            encoder = TransformerEncoder(d_model=hidden_size, num_heads=heads, d_ff=ff_size, rate=dropout,
                                         num_layers=layer_num)

            sents_vec = plm(inputs=plm_inputs)
            sents_vec = encoder(x=sents_vec, training=True, mask=sentence_masks)
            quality_dense = QualityDense(encoder=encoder)
            quality_logits = quality_dense(sents_vec)

        quality_softmax = QualitySoftmax()
        softmax_inputs = {
            "quality_logits": quality_logits,
            "quality_labels": quality_labels
        }
        quality_probs = quality_softmax(softmax_inputs)

        outputs = quality_probs
        opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, lr=2e-5)

        # warm up opt
        learning_rate = CustomSchedule(d_model=768, warmup_steps=4000)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dialogue_quality")
        model.compile(optimizer=opt, run_eagerly=True)

        return model
