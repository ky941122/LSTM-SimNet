import tensorflow as tf


class SimNet():
    def __init__(self, sequence_length, hidden_size, is_training, dropout_keep_prob, vocab_size, embedding_size, fc_size,
                 num_classes=2):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_query")
        self.seq_len_1 = tf.placeholder(tf.int32, [None], name="input_query_length")

        self.input_x_2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_candidate")
        self.seq_len_2 = tf.placeholder(tf.int32, [None], name="input_candidate_length")

        self.label = tf.placeholder(tf.float32, [None, num_classes], name="input_label")

        with tf.variable_scope("Embedding"):
            self._embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size])

            inputs_que = tf.nn.embedding_lookup(self._embedding, self.input_x_1)
            inputs_can = tf.nn.embedding_lookup(self._embedding, self.input_x_2)

            if self.is_training and self.dropout_keep_prob < 1:
                inputs_que = tf.nn.dropout(inputs_que, self.dropout_keep_prob)
                inputs_can = tf.nn.dropout(inputs_can, self.dropout_keep_prob)

        with tf.variable_scope("RNN"):
            cell_type = tf.contrib.rnn.LSTMCell

            self.fw_cell = cell_type(
                num_units=self.hidden_size, state_is_tuple=True)
            self.bw_cell = cell_type(
                num_units=self.hidden_size, state_is_tuple=True)

            que_outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell,
                                    inputs_que, sequence_length=self.seq_len_1, dtype=tf.float32)
            que_encoder = tf.concat(que_outputs, -1)
            que_rep = self.extract_last_step(que_encoder, self.seq_len_1)


            can_outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell,
                                    inputs_can, sequence_length=self.seq_len_2, dtype=tf.float32)
            can_encoder = tf.concat(can_outputs, -1)
            can_rep = self.extract_last_step(can_encoder, self.seq_len_2)

        with tf.variable_scope("Fully_Connection"):
            w = tf.get_variable("W", [self.hidden_size*4, self.fc_size])
            b = tf.get_variable("b", [self.fc_size])

            rep_concat = tf.concat([que_rep, can_rep], -1)
            out_without_bias = tf.matmul(rep_concat, w)
            fc_out = tf.nn.bias_add(out_without_bias, b)
            hidden_relu = tf.nn.relu(fc_out)

        with tf.variable_scope("Output"):
            W = tf.get_variable("W", [self.fc_size, self.num_classes])
            bias = tf.get_variable("b", [self.num_classes])

            output = tf.matmul(hidden_relu, W)
            self.logits = tf.nn.bias_add(output, bias, name="logits")

        self.prob = tf.nn.softmax(self.logits, -1, name="probs")

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                      labels=self.label))

        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, "float"))

        self._model_stats()


    def extract_last_step(self, input_hidden, seq_length):
        output = input_hidden
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (seq_length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant




    @staticmethod
    def _model_stats():
        """Print trainable variables and total model size."""

        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())

        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))


