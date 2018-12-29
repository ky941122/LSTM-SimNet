#coding=utf-8
from __future__ import division

import os
import time

import numpy as np
import tensorflow as tf
#from tensorflow.python import debug as tf_debug

from simnet import SimNet
import data_loader
import datetime



# Data
tf.flags.DEFINE_string("train_file", "data/id_train", "train data (id)")
tf.flags.DEFINE_string("dev_data", "data/id_test", "dev data (id)")
tf.flags.DEFINE_integer("vocab_size", 19542, "vocab.txt")
tf.flags.DEFINE_integer("pad_id", 0, "id for <pad> token in character list")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length", 20, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_size", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("rnn_hidden_size", 512, "rnn")
tf.flags.DEFINE_integer("fc_hidden_size", 128, "fc")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning_rate (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("max_epoch", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 5000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Save Model
tf.flags.DEFINE_string("model_name", "PointwiseSimNet", "model name")
tf.flags.DEFINE_integer("num_checkpoints", 2000, "checkpoints number to save")
tf.flags.DEFINE_boolean("restore_model", False, "Whether restore model or create new parameters")
tf.flags.DEFINE_string("model_path", "runs", "Restore which model")
tf.flags.DEFINE_boolean("restore_pretrained_embedding", False, "Whether restore pretrained embedding")
tf.flags.DEFINE_string("pretrained_embeddings_path", "checkpoints/embedding", "Restore pretrained embedding")




FLAGS = tf.flags.FLAGS


def print_args(flags):
    """Print arguments."""
    print("\nParameters:")
    for attr in flags:
        value = flags[attr].value
        print("{}={}".format(attr, value))
    print("")


def train():
    print "Loading data..."
    y, q, lq, c, lc = data_loader.read_data(FLAGS.train_file, FLAGS.max_sequence_length, FLAGS.pad_id)
    y_test, q_test, lq_test, c_test, lc_test = data_loader.read_data(FLAGS.dev_data, FLAGS.max_sequence_length, FLAGS.pad_id)
    print "Data Size:", len(y)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        with sess.as_default():
            model = SimNet(sequence_length=FLAGS.max_sequence_length,
                           hidden_size=FLAGS.rnn_hidden_size,
                           is_training=True,
                           dropout_keep_prob=FLAGS.dropout_keep_prob,
                           vocab_size=FLAGS.vocab_size,
                           embedding_size=FLAGS.embedding_size,
                           fc_size=FLAGS.fc_hidden_size
                           )

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            capped_gvs = [(tf.clip_by_value(grad, -3., 3.), var) for grad, var in grads_and_vars]

            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.model_name, timestamp))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            ############restore embedding##################
            if FLAGS.restore_pretrained_embedding:
                embedding_var_name = "Embedding/embedding:0"

                # 得到该网络中，所有可以加载的参数
                variables = tf.contrib.framework.get_variables_to_restore()

                variables_to_resotre = [v for v in variables if v.name == embedding_var_name]

                saver = tf.train.Saver(variables_to_resotre)

                saver.restore(sess, FLAGS.pretrained_embeddings_path)
                print "Restore embeddings from", FLAGS.pretrained_embeddings_path

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            restore = FLAGS.restore_model
            if restore:
                saver.restore(sess, FLAGS.model_path)
                print("*" * 20 + "\nReading model parameters from %s \n" % FLAGS.model_path + "*" * 20)
            else:
                print("*" * 20 + "\nCreated model with fresh parameters.\n" + "*" * 20)


            def train_step(y, q, lq, c, lc, epoch):

                """
                A single training step
                """

                lq = np.reshape(lq, [-1])
                lc = np.reshape(lc, [-1])

                feed_dict = {
                    model.input_x_1: q,
                    model.input_x_2: c,
                    model.label: y,
                    model.seq_len_1: lq,
                    model.seq_len_2: lc
                }

                _, step, loss, acc = sess.run([train_op, global_step, model.loss, model.accuracy], feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print "{}: Epoch {} step {}, loss {:g}, accuracy {:g}".format(time_str, epoch, step, loss, acc)



            def dev_step(y, q, lq, c, lc):

                lq = np.reshape(lq, [-1])
                lc = np.reshape(lc, [-1])

                feed_dict = {
                    model.input_x_1: q,
                    model.input_x_2: c,
                    model.label: y,
                    model.seq_len_1: lq,
                    model.seq_len_2: lc
                }

                correct_num = sess.run(model.correct_num, feed_dict)

                return correct_num



            # Generate batches
            batches = data_loader.batch_iter(list(zip(y, q, lq, c, lc)),
                                             FLAGS.batch_size, FLAGS.max_epoch, True)

            num_batches_per_epoch = int((len(y)) / FLAGS.batch_size) + 1

            # Training loop. For each batch...
            epoch = 0
            max_acc = 0
            max_acc_step = 0
            for batch in batches:
                y_batch, q_batch, lq_batch, c_batch, lc_batch = zip(*batch)

                train_step(y_batch, q_batch, lq_batch, c_batch, lc_batch, epoch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % num_batches_per_epoch == 0:
                    epoch += 1

                if current_step % FLAGS.evaluate_every == 0:
                    batches_test = data_loader.batch_iter(list(zip(y_test, q_test, lq_test, c_test, lc_test)),
                                                          FLAGS.batch_size, 1, False)

                    total_num = 0
                    for test_batch in batches_test:
                        y_batch_test, q_batch_test, lq_batch_test, c_batch_test, lc_batch_test = zip(*test_batch)

                        num_this_batch = dev_step(y_batch_test, q_batch_test, lq_batch_test, c_batch_test, lc_batch_test)

                        total_num += num_this_batch

                    acc = total_num / len(y_test)

                    print "{} step evaluation result: {}".format(current_step, acc)

                    if acc > max_acc:
                        max_acc = acc
                        max_acc_step = current_step

                    print "until now, the max accuracy is {}, in {} step.".format(max_acc, max_acc_step)


                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))



if __name__ == "__main__":
    print_args(FLAGS)

    train()

