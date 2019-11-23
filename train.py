#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math, random
import os, time
import numpy as np
#from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
import os

import data_utils
# from data_utils import *
import argparse
from model import TCVAE
import collections
from gensim.models import KeyedVectors

import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, \
                    format='%(asctime)s %(module)s %(levelname)-8s %(message)s', \
                    datefmt='%Y-%m-%d %H:%M:%S', \
                    handlers=[
                        logging.FileHandler("t-cvae.log"),
                        logging.StreamHandler()
                    ])
FLAGS = None

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # folders
    parser.add_argument("--data_dir", type=str, default="data/", help="Data directory")
    parser.add_argument('--dataset', type=str, default='mscoco', help='which dataset to use')
    parser.add_argument("--model_dir", type=str, default="model/", help="Model directory")
    parser.add_argument("--output_dir", type=str, default="output/", help="Out directory")
    # gpu_device
    parser.add_argument("--gpu_device", type=str, default="0", help="which gpu to use")
    # data files
    parser.add_argument("--train_data", type=str, default="/train.ids",
                        help="Training data path")
    parser.add_argument("--valid_data", type=str, default="/valid.ids",
                        help="Valid data path")
    parser.add_argument("--test_data", type=str, default="/test.ids",
                        help="Test data path")
    # vocab
    parser.add_argument("--from_vocab", type=str, default="/vocab_20000",
                        help="from vocab path")
    parser.add_argument("--from_vocab_size", type=int, default=20000, help="vocab size")
    parser.add_argument("--to_vocab_size", type=int, default=20000)
    parser.add_argument("--to_vocab", type=str, default="/vocab_20000",
                        help="to vocab path")
    
    parser.add_argument("--max_train_data_size", type=int, default=0, help=\
                        "Limit on the size of training data (0: no limit)")
    # modeal and training
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--num_units", type=int, default=256, help="Size of each model layer")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads in attention")
    parser.add_argument("--emb_dim", type=int, default=300, help="Dimension of word embedding")
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of latent variable")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use during training")
    parser.add_argument("--max_gradient_norm", type=float, default=3.0, help="Clip gradients to this norm")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.5, \
                        help="Learning rate decays by this much")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--epoch_num", type=int, default=100, help="Number of epoch")


def create_args(flags):
    return tf.contrib.training.HParams(
        # dir path
        data_dir=flags.data_dir,
        model_dir=flags.model_dir,
        output_dir=flags.output_dir,
        dataset = flags.dataset,
        # data params
        batch_size=flags.batch_size,
        from_vocab_size=flags.from_vocab_size,
        to_vocab_size=flags.to_vocab_size,
        GO_ID=data_utils.GO_ID,
        EOS_ID=data_utils.EOS_ID,
        PAD_ID=data_utils.PAD_ID,

        train_data=flags.train_data,
        valid_data=flags.valid_data,
        test_data=flags.test_data,

        from_vocab=flags.data_dir+flags.dataset+flags.from_vocab,
        to_vocab=flags.data_dir+flags.dataset+flags.to_vocab,

        dropout_rate=flags.dropout_rate,
        init_weight=0.1,
        emb_dim=flags.emb_dim,
        latent_dim=flags.latent_dim,
        num_units=flags.num_units,
        num_heads=flags.num_heads,
        num_layers=flags.num_layers,
        learning_rate=flags.learning_rate,
        clip_value=flags.max_gradient_norm,
        decay_factor=flags.learning_rate_decay_factor,
        epoch_num=flags.epoch_num,
    )

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True
    return config_proto

class TrainModel(
    collections.namedtuple("TrainModel",
                           ("graph", "model"))):
    pass

class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model"))):
    pass

class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model"))):
    pass

def create_model(args, model):
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_model = model(args, tf.contrib.learn.ModeKeys.TRAIN)

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        eval_model = model(args, tf.contrib.learn.ModeKeys.EVAL)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_model = model(args, tf.contrib.learn.ModeKeys.INFER)

    return TrainModel(graph=train_graph, model=train_model), \
           EvalModel(graph=eval_graph, model=eval_model), \
           InferModel(graph=infer_graph, model=infer_model)

def read_data(src_path):
    data_set = []
    cnt = 0
    max_stn_length = 0
    with tf.gfile.GFile(src_path, mode="r") as src_file:
        src = src_file.readline()
        while src:
            if cnt % 100000 == 0:
                logging.info("reading data line %d" % cnt)
            sample, sentence = [], []
            token_idx_str_list = src.split(' ')
            for token_idx_str in token_idx_str_list:
                token_idx = int(token_idx_str)
                if token_idx != -1:
                    sentence.append(token_idx)
                else:
                    max_stn_length = max(max_stn_length, len(sentence))
                    # truncate all sentence to a maximum length of 25
                    sample.append(sentence[:25])
                    sentence = []
            data_set.append(sample)
            cnt += 1
            src = src_file.readline()
    logging.info('sample count: %d' % cnt)
    logging.info('max length of single sentence: %d' % max_stn_length)
    return data_set

def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans

def train(args):
    word_embs = init_embedding(args)
    logging.info("vocab and word embeddings load over.")
    args.add_hparam(name="embeddings", value=word_embs)
    train_model, eval_model, infer_model = create_model(args, TCVAE)
    config = get_config_proto(log_device_placement=False)
    train_sess = tf.Session(config=config, graph=train_model.graph)
    eval_sess = tf.Session(config=config, graph=eval_model.graph)
    infer_sess = tf.Session(config=config, graph=infer_model.graph)
    logging.info("model create over.")
    train_data = read_data(args.data_dir + args.dataset + args.train_data)
    valid_data = read_data(args.data_dir + args.dataset + args.valid_data)
    test_data = read_data(args.data_dir + args.dataset + args.test_data)

    ckpt = tf.train.get_checkpoint_state(args.model_dir+args.dataset)
    ckpt_path = os.path.join(args.model_dir, "ckpt")
    with train_model.graph.as_default():
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            train_model.model.saver.restore(train_sess, ckpt.model_checkpoint_path)
            eval_model.model.saver.restore(eval_sess, ckpt.model_checkpoint_path)
            infer_model.model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
            global_step = train_model.model.global_step.eval(session=train_sess)
        else:
            train_sess.run(tf.global_variables_initializer())
            global_step = 0
    to_vocab, rev_to_vocab = data_utils.initialize_vocabulary(args.from_vocab)

    step_loss, step_time, total_predict_count, total_loss, total_time, avg_loss, avg_time = \
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


    while global_step <= 380000:
        start_time = time.time()
        step_loss, global_step, predict_count = train_model.model.train_step(train_sess, train_data)

        total_loss += step_loss / args.batch_size
        total_time += (time.time() - start_time)
        total_predict_count += predict_count
        if global_step % 100 == 0:
            ppl = safe_exp(total_loss * args.batch_size / total_predict_count)
            avg_loss = total_loss / 100
            avg_time = total_time / 100
            total_loss, total_predict_count, total_time = 0.0, 0.0, 0.0
            logging.info("global step %d   step-time %.2fs  loss %.3f  ppl %.2f" % \
                        (global_step, avg_time, avg_loss, ppl))

        if  global_step % 3000 == 0:
            train_model.model.saver.save(train_sess, ckpt_path, global_step=global_step)
            ckpt = tf.train.get_checkpoint_state(args.model_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                eval_model.model.saver.restore(eval_sess, ckpt.model_checkpoint_path)
                infer_model.model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
                logging.info("load eval model.")
            else:
                raise ValueError("ckpt file not found.")
            for batch_idx in range(0, int(len(valid_data)/args.batch_size)):
                step_loss, predict_count = eval_model.model.eval_step(eval_sess, valid_data, no_random=True, \
                                                                      batch_idx=batch_idx * args.batch_size)
                total_loss += step_loss
                total_predict_count += predict_count
            ppl = safe_exp(total_loss / total_predict_count)

            total_loss, total_predict_count, total_time = 0.0, 0.0, 0.0
            logging.info("eval ppl %.2f" % (ppl))
            if global_step < 30000:
                continue
            logging.info('begin infer')
            f1 = open("output/" + args.dataset + "/ref2_file" + str(global_step),"w",encoding="utf-8")
            f2 = open("output/" + args.dataset + "/predict2_file" + str(global_step),"w", encoding="utf-8")
            for batch_idx in range(0, int(len(valid_data) / args.batch_size)):
                given, answer, predict = infer_model.model.infer_step(infer_sess, test_data, no_random=True,
                                                                      batch_idx=batch_idx * args.batch_size)
                for i in range(args.batch_size):
                    sample_output = predict[i]
                    if args.EOS_ID in sample_output:
                        sample_output = sample_output[:sample_output.index(args.EOS_ID)]
                    pred = []
                    for output in sample_output:
                        pred.append(tf.compat.as_str(rev_to_vocab[output]))

                    sample_output = answer[i]
                    if args.EOS_ID in sample_output[:]:
                        if sample_output[0] == args.GO_ID:
                            sample_output = sample_output[1:sample_output.index(args.EOS_ID)]
                        else:
                            sample_output = sample_output[0:sample_output.index(args.EOS_ID)]
                    ans = []
                    for output in sample_output:
                        ans.append(tf.compat.as_str(rev_to_vocab[output]))
                    if batch_idx == 0 and i < 8:
                        print("answer: ", " ".join(ans))
                        print("predict: ", " ".join(pred))

                    f1.write(" ".join(ans).replace("_UNK", "_unknown") + "\n")
                    f2.write(" ".join(pred) + "\n")
            f1.close()
            f2.close()
            hyp_file = "output/" + args.dataset + "/predict2_file" + str(global_step)
            ref_file = "output/" + args.dataset + "/ref2_file" + str(global_step)
            result = os.popen("python multi_bleu.py -ref " + ref_file + " -hyp " + hyp_file)
            logging.info(result.read())

            f3 = open(hyp_file, "r", encoding="utf-8")
            dic1 = {}
            dic2 = {}
            distinc1, distinc2 = 0, 0
            all1, all2 = 0, 0
            t = 0
            for l in f3:
                line = l.rstrip("\n").split(" ")
                for word in line:
                    all1 += 1
                    if word not in dic1:
                        dic1[word] = 1
                        distinc1 += 1
                for i in range(0, len(line) - 1):
                    all2 += 1
                    if line[i] + " " + line[i + 1] not in dic2:
                        dic2[line[i] + " " + line[i + 1]] = 1
                        distinc2 += 1
            logging.info("distinc1: %.5f" % float(distinc1 / all1))
            logging.info("distinc2: %.5f" % float(distinc2 / all2))
            logging.info("*"*32 + "infer done." + "*"*32)

def init_embedding(args):
    vocab = []
    with open(args.from_vocab, "r", encoding="utf-8") as f:
        for line in f:
            vocab.append(line.rstrip("\n"))
    word_vectors = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)
    # word_vectors = KeyedVectors.load_word2vec_format("roc_vector.txt")
    # word_vectors = KeyedVectors.load_word2vec_format("data/glove.42B.300d.txt")
    # model = Word2Vec(sentences=sent, sg=1, size=256, window=5, min_count=3, hs=1)
    # model.save("word2vec")
    word_embs = []
    num_of_words_with_emb = 0
    for i in range(0, len(vocab)):
        word = vocab[i]
        if word in word_vectors:
            num_of_words_with_emb += 1
            word_embs.append(word_vectors[word])
        else:
            word_embs.append((0.1 * np.random.random([args.emb_dim]) - 0.05).astype(np.float32))
    logging.info("load word embeddings finished")
    word_embs = np.array(word_embs)
    logging.info('num of words with embedding: %d' % num_of_words_with_emb)
    logging.info('embedding matrix shape: %s' % str(word_embs.shape))
    return word_embs

def main(_):
    hparams = create_args(FLAGS)
    train(hparams)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    add_arguments(my_parser)
    FLAGS, remaining = my_parser.parse_known_args()
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dataset
    FLAGS.output_dir = FLAGS.output_dir + FLAGS.dataset
    logging.info(FLAGS)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device
    tf.app.run()