from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py
import collections
#from python_algorithms.basic.union_find import UF
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.cm as cm



import util
import coref_ops
import conll
import metrics
import evaluators

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def sparse2dense(x):
  return tf.sparse_to_dense(x.indices, x.dense_shape, x.values)

class CorefModel(object):
  #def __init__(self, config):
  def __init__(self, config, session=None):
    self.config = config
    self.session = session
    self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
    self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
    self.char_embedding_size = config["char_embedding_size"]
    self.char_dict = util.load_char_dict(config["char_vocab_path"])
    self.max_span_width = config["max_span_width"]
    self.singleton_thresh = config["singleton_thresh"]
    self.genres = { g:i for i,g in enumerate(config["genres"]) }
    self.plural_clusters = config["plural_clusters"]
    self.singularity = config["singularity"] # 0 < x < 1; 1 => no plural pruning
    self.many_antecedents = config["many_antecedents"]
    self.ln = config["ln"]
    self.L = config["l"]
    self.R = config["r"]
    self.comb = config["comb"]
    if self.comb: self.plural_clusters = True
    if config["lm_path"]:
      try:
        self.lm_file = h5py.File(self.config["lm_path"], "r")
      except:
        print("Couldn't load LM", self.config['lm_path'])
        self.lm_file = None
    else:
      self.lm_file = None
    self.lm_layers = self.config["lm_layers"]
    self.lm_size = self.config["lm_size"]
    self.eval_data = None # Load eval data lazily.
    self.debug = self.config["debug"] # collections.OrderedDict()
    self.spk = self.config["spk"]
    if self.spk:
      with open("speaker_file.txt",'rb') as fp:
        spks = pickle.load(fp)
      # self.spk_emb = tf.random_uniform([len(spks) + 1, self.config["feature_size"]])
      #self.spk_emb = np.random.uniform(size = (len(spks) + 1, self.config["feature_size"]))
      self.spk_dict = {name: i + 2 for i, name in enumerate(spks)}
      #print('self.spk_dict', len(self.spk_dict))
      #print('self.spk_emb', self.spk_emb.shape)

    input_props = []
    input_props.append((tf.string, [None, None])) # Tokens.
    input_props.append((tf.float32, [None, None, self.context_embeddings.size])) # Context embeddings.
    input_props.append((tf.float32, [None, None, self.head_embeddings.size])) # Head embeddings.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers])) # LM embeddings.
    input_props.append((tf.int32, [None, None, None])) # Character indices.
    input_props.append((tf.int32, [None])) # Text lengths.
    #input_props.append((tf.int32, [None])) # Speaker IDs.
    if False and self.spk:
      input_props.append((tf.float32, [None, None]))  # Speaker IDs.
    else:
      input_props.append((tf.int32, [None, None] if self.config["plural_speakers"] else [None]))  # Speaker IDs.
    input_props.append((tf.int32, [])) # Genre.
    input_props.append((tf.bool, [])) # Is training.
    input_props.append((tf.int32, [None])) # Gold starts.
    input_props.append((tf.int32, [None])) # Gold ends.
    input_props.append((tf.int32, [None, None] if self.plural_clusters else [None])) # Cluster ids.

    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    #print('*self.input_tensors', *self.input_tensors)
    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)
    learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               self.config["decay_frequency"], self.config["decay_rate"], staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam" : tf.train.AdamOptimizer,
      "sgd" : tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

  def start_enqueue_thread(self, session):
    with open(self.config["train_path"]) as f:
      train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
    def _enqueue_loop():
      while True:
        random.shuffle(train_examples)
        for example in train_examples:
          tensorized_example = self.tensorize_example(example, is_training=True)
          feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
          #print('self.enqueue_op',self.enqueue_op)
          #print('feed_dict',feed_dict)
          session.run(self.enqueue_op, feed_dict=feed_dict)
          #exit()
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def restore(self, session):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
    saver = tf.train.Saver(vars_to_restore)
    checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)

  def load_lm_embeddings(self, doc_key):
    if self.lm_file is None:
      return np.zeros([0, 0, self.lm_size, self.lm_layers])
    file_key = doc_key.replace("/", ":")
    group = self.lm_file[file_key]
    num_sentences = len(list(group.keys()))
    sentences = [group[str(i)][...] for i in range(num_sentences)]
    lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
    for i, s in enumerate(sentences):
      lm_emb[i, :s.shape[0], :, :] = s
    return lm_emb

  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = zip(*mentions)
    else:
      starts, ends = [], []
    assert(len(starts) == len(ends))
    return np.array(starts), np.array(ends)

  def tensorize_span_labels(self, tuples, label_dict):
    if len(tuples) > 0:
      starts, ends, labels = zip(*tuples)
    else:
      starts, ends, labels = [], [], []
    return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

  def tensorize_example(self, example, is_training):
    clusters = example["clusters"]
    #print("TENSORIZE-----------")
    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}



    # todo clusters
    if self.plural_clusters:
      cluster_list = collections.defaultdict(list) # c_l[mention] = [c0 c1 ... cn]
      #max_cluster_length = 0
      for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
          cluster_list[gold_mention_map[tuple(mention)]].append(cluster_id + 1)
      max_cluster_length = max([1] + [len(l) for l in cluster_list.values()]) if cluster_list else 1
      #assert(max_cluster_length <= 6)
      cluster_ids = np.zeros((len(gold_mentions), max_cluster_length))
      for mention, clusters in cluster_list.items():
        for cluster_id, cluster in enumerate(clusters):
          cluster_ids[mention][cluster_id] = cluster
    else:
      cluster_ids = np.zeros(len(gold_mentions))
      for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
          cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1
    assert(len(cluster_ids.shape) == 2 if self.plural_clusters else 1)

    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)
    speakers = util.flatten(example["speakers"])

    assert num_words == len(speakers)
    max_sentence_length = max(len(s) for s in sentences)
    max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
    text_len = np.array([len(s) for s in sentences])
    tokens = [[""] * max_sentence_length for _ in sentences]
    context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
    head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
    char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
    for i, sentence in enumerate(sentences):
      for j, word in enumerate(sentence):
        tokens[i][j] = word
        context_word_emb[i, j] = self.context_embeddings[word]
        head_word_emb[i, j] = self.head_embeddings[word]
        char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
    tokens = np.array(tokens)

    # todo speakers
    #print('speakers')
    if self.config["plural_speakers"]:
      #speaker_list = [list(set(speaker_group.split('|'))) for speaker_group in speakers]
      speaker_list = [list(set(speaker_group.split('|'))) for speaker_group in speakers]
      max_speaker_group = max(map(len, speaker_list))
      if self.spk:
        ids = [[self.spk_dict.get(speaker, 1) for speaker in group] for group in speaker_list] # [sent][spks]
        #speaker_ids = tf.gather(self.spk_emb, tf.convert_to_tensor(list(ids))) # [sent][spks][emb]
        #speaker_ids = tf.reduce_mean(speaker_ids, 1) # [sent][emb]
        #print('speaker_list', speaker_list)
        #print('ids', ids)
        #speaker_ids = np.take(self.spk_emb, ids)
        #print('speaker_ids',speaker_ids.shape, speaker_ids)
        #speaker_ids = np.mean(speaker_ids, axis=1)
        #speaker_ids = np.array(ids)
        speaker_ids = np.array([[s for s in group] + [0 for _ in range(max_speaker_group - len(group))]
                                for group in ids])
        #print('speaker_ids',speaker_ids.shape, speaker_ids)
      else:
        speaker_dict = {s: i + 1 for i, s in enumerate(set(sp for group in speaker_list for sp in group))}
        speaker_ids = np.array([[speaker_dict[s] for s in group] + [0 for _ in range(max_speaker_group - len(group))]
                                for group in speaker_list])
    else:
      speaker_dict =  { s:i for i,s in enumerate(set(speakers)) }
      speaker_ids = np.array([speaker_dict[s] for s in speakers])

    doc_key = example["doc_key"]
    genre = self.genres[doc_key[:2]] if self.config["use_genre"] else self.genres['nw']

    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

    lm_emb = self.load_lm_embeddings(doc_key)

    example_tensors = (tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids)

    if is_training and len(sentences) > self.config["max_training_sentences"]:
      return self.truncate_example(*example_tensors)
    else:
      return example_tensors

  def truncate_example(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):
    max_training_sentences = self.config["max_training_sentences"]
    num_sentences = context_word_emb.shape[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences)
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    tokens = tokens[sentence_offset:sentence_offset + max_training_sentences, :]
    context_word_emb = context_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
    head_word_emb = head_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
    lm_emb = lm_emb[sentence_offset:sentence_offset + max_training_sentences, :, :, :]
    char_index = char_index[sentence_offset:sentence_offset + max_training_sentences, :, :]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

    speaker_ids = speaker_ids[word_offset: word_offset + num_words]
    gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset
    cluster_ids = cluster_ids[gold_spans]

    return tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids

  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
    same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0)) # [num_labeled, num_candidates]
    same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) # [num_labeled, num_candidates]
    same_span = tf.logical_and(same_start, same_end) # [num_labeled, num_candidates]
    if self.plural_clusters:
      candidate_labels = tf.matmul(labels, tf.to_int32(same_span), transpose_a=True)  # [max_cluster, num_candidates]
      candidate_labels = tf.transpose(candidate_labels) # [num_candidates, max_cluster]
    else:
      candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span)) # [1, num_candidates]
      candidate_labels = tf.squeeze(candidate_labels, 0) # [num_candidates]
    return candidate_labels

  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)

  def plural_coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
    k = util.shape(top_span_emb, 0)
    top_span_range = tf.range(k)  # [k]
    antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0)  # [k, k]
    antecedents_mask = antecedent_offsets >= 1  # [k, k]
    # antecedents_mask = antecedent_offsets >= 0  # [k, k]
    fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores,
                                                                                         0)  # [k, k]
    fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask))  # [k, k]
    fast_antecedent_scores += self.get_plural_fast_antecedent_scores(top_span_emb)  # [k, k]

    _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False)  # [k, c]
    top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents)  # [k, c]
    top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents)  # [k, c]
    top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents)  # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
    k = util.shape(top_span_emb, 0)
    top_span_range = tf.range(k) # [k]
    antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0) # [k, k]
    #todo add offset >=0 for singletons
    antecedents_mask = antecedent_offsets >= 1 # [k, k]
    #antecedents_mask = antecedent_offsets >= 0  # [k, k]
    fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0) # [k, k]
    fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask)) # [k, k]
    fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb) # [k, k]

    _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False) # [k, c]
    top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents) # [k, c]
    top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents) # [k, c]
    top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def distance_pruning(self, top_span_emb, top_span_mention_scores, c):
    k = util.shape(top_span_emb, 0)
    top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1]) # [k, c]
    raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets # [k, c]
    top_antecedents_mask = raw_top_antecedents >= 0 # [k, c]
    top_antecedents = tf.maximum(raw_top_antecedents, 0) # [k, c]

    top_fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores, top_antecedents) # [k, c]
    top_fast_antecedent_scores += tf.log(tf.to_float(top_antecedents_mask)) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def get_predictions_and_loss(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):
    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
    self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
    self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

    num_sentences = tf.shape(context_word_emb)[0]
    max_sentence_length = tf.shape(context_word_emb)[1]

    context_emb_list = [context_word_emb]
    head_emb_list = [head_word_emb]

    if self.config["char_embedding_size"] > 0:
      char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), char_index) # [num_sentences, max_sentence_length, max_word_length, emb]
      flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
      flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
      aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
      context_emb_list.append(aggregated_char_emb)
      head_emb_list.append(aggregated_char_emb)

    if not self.lm_file:
      elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
      lm_embeddings = elmo_module(
          inputs={"tokens": tokens, "sequence_len": text_len},
          signature="tokens", as_dict=True)
      word_emb = lm_embeddings["word_emb"]  # [num_sentences, max_sentence_length, 512]
      lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                         lm_embeddings["lstm_outputs1"],
                         lm_embeddings["lstm_outputs2"]], -1)  # [num_sentences, max_sentence_length, 1024, 3]
    lm_emb_size = util.shape(lm_emb, 2)
    lm_num_layers = util.shape(lm_emb, 3)
    with tf.variable_scope("lm_aggregation"):
      self.lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
      self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
    flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
    flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights, 1)) # [num_sentences * max_sentence_length * emb, 1]
    aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
    aggregated_lm_emb *= self.lm_scaling
    context_emb_list.append(aggregated_lm_emb)

    context_emb = tf.concat(context_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    head_emb = tf.concat(head_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    context_emb = tf.nn.dropout(context_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]
    head_emb = tf.nn.dropout(head_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]

    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length) # [num_sentence, max_sentence_length]

    context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask) # [num_words, emb]
    num_words = util.shape(context_outputs, 0)

    genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]), genre) # [emb]

    sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]
    flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask) # [num_words]
    flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask) # [num_words]

    candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width]) # [num_words, max_span_width]
    candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0) # [num_words, max_span_width]
    candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) # [num_words, max_span_width]
    candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) # [num_words, max_span_width]
    candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) # [num_words, max_span_width]
    flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]
    candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask) # [num_candidates]
    candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask) # [num_candidates]
    candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]), flattened_candidate_mask) # [num_candidates]

    #todo cluster
    candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids) # [num_candidates]

    candidate_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, candidate_starts, candidate_ends) # [num_candidates, emb]
    candidate_mention_scores =  self.get_mention_scores(candidate_span_emb) # [k, 1]
    candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [k]

    k = tf.to_int32(tf.floor(tf.to_float(tf.shape(context_outputs)[0]) * self.config["top_span_ratio"]))
    top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                               tf.expand_dims(candidate_starts, 0),
                                               tf.expand_dims(candidate_ends, 0),
                                               tf.expand_dims(k, 0),
                                               util.shape(context_outputs, 0),
                                               True) # [1, k]
    top_span_indices.set_shape([1, None])
    top_span_indices = tf.squeeze(top_span_indices, 0) # [k]

    top_span_starts = tf.gather(candidate_starts, top_span_indices) # [k]
    top_span_ends = tf.gather(candidate_ends, top_span_indices) # [k]
    top_span_emb = tf.gather(candidate_span_emb, top_span_indices) # [k, emb]
    #ptop_span_emb = top_span_emb  # [k, emb]
    top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices) # [k]
    top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices) # [k]
    top_span_sentence_indices = tf.gather(candidate_sentence_indices, top_span_indices) # [k]

    with tf.control_dependencies([tf.assert_rank(speaker_ids, 2 if self.config["plural_speakers"] else 1)]):
      top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts)  # [k]

    c = tf.minimum(self.config["max_top_antecedents"], k)

    if self.config["coarse_to_fine"]:
      top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c)
      if self.plural_clusters:
        ptop_antecedents, ptop_antecedents_mask, ptop_fast_antecedent_scores, ptop_antecedent_offsets = self.plural_coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c)
    else:
      top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.distance_pruning(top_span_emb, top_span_mention_scores, c)
      if self.plural_clusters:
        ptop_antecedents, ptop_antecedents_mask, ptop_fast_antecedent_scores, ptop_antecedent_offsets = self.distance_pruning(
          top_span_emb, top_span_mention_scores, c)

    dummy_scores = tf.zeros([k, 1]) # [k, 1]
    for i in range(self.config["coref_depth"]):
      with tf.variable_scope("coref_layer", reuse=tf.AUTO_REUSE):  # (i > 0 or self.plural_clusters)):
        if self.plural_clusters:
          ptop_antecedent_emb = tf.gather(top_span_emb, ptop_antecedents)  # [k, c, emb]
          ptop_antecedent_scores = ptop_fast_antecedent_scores + self.get_plural_slow_antecedent_scores(top_span_emb, ptop_antecedents, ptop_antecedent_emb, ptop_antecedent_offsets, top_span_speaker_ids, genre_emb) # [k, c]
          ptop_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, ptop_antecedent_scores], 1))  # [k, c + 1]
          ptop_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), ptop_antecedent_emb], 1)  # [k, c + 1, emb]
          pattended_span_emb = tf.reduce_sum(tf.expand_dims(ptop_antecedent_weights, 2) * ptop_antecedent_emb, 1)  # [k, emb]

        top_antecedent_emb = tf.gather(top_span_emb, top_antecedents) # [k, c, emb]
        top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, genre_emb) # [k, c]
        top_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, top_antecedent_scores], 1)) # [k, c + 1]
        top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb], 1) # [k, c + 1, emb]
        attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb, 1) # [k, emb]
        with tf.variable_scope("f"):
          old_top_span_emb = top_span_emb
          f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1), util.shape(top_span_emb, -1))) # [k, emb]
          top_span_emb = f * attended_span_emb + (1 - f) * old_top_span_emb  # [k, emb]
          if self.plural_clusters:
            with tf.variable_scope("plural"):
              pf = tf.sigmoid(util.projection(tf.concat([top_span_emb, pattended_span_emb], 1), util.shape(top_span_emb, -1)))  # [k, emb]
              top_span_emb = self.singularity * top_span_emb + (1-self.singularity) * (pf * pattended_span_emb + (1 - pf) * old_top_span_emb)  # [k, emb]


    top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1) # [k, c + 1]
    top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents) # [k, c]

    if self.plural_clusters:
      max_clusters = util.shape(top_span_cluster_ids, 1)
      span_groups = tf.count_nonzero(top_span_cluster_ids, 1)  # tf.logical_and(groups >= 1, groups <= 1)
      span_groups = tf.tile(tf.expand_dims(span_groups, 1), [1, c])  # [k, c]
      span_singular = span_groups <= 1
      non_dummy_indicator = span_groups > 0  # [k, c]
      span_plural = span_groups >= 2  # at least in 2 clusters

      # L: mi is sing => ant is sing
      top_antecedents_mask = tf.tile(tf.expand_dims(top_antecedents_mask, 2), [1, 1, max_clusters]) # [k, c, clu]
      top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask)))  # [k, c, clu]
      expanded_span_cluster_ids = tf.tile(tf.expand_dims(top_span_cluster_ids, 1), [1, c, 1])
      same_cluster_indicator = tf.sets.set_intersection(top_antecedent_cluster_ids, expanded_span_cluster_ids)
      same_cluster_indicator = sparse2dense(same_cluster_indicator)  # tf.sparse_to_dense()
      same_cluster_indicator = tf.count_nonzero(same_cluster_indicator, 2) > 0  # [k, c]

      ant_singular = tf.count_nonzero(top_antecedent_cluster_ids, 2) <= 1 # [k, c]
      valid_indicator = tf.logical_and(non_dummy_indicator, ant_singular if self.ln else span_singular)
      pairwise_labels = tf.logical_and(same_cluster_indicator, valid_indicator)  # [k, c]
      dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
      top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]
      loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)  # [k]
      loss = tf.reduce_sum(loss)  # []

      # R: mi is plur and mj is sing => ant is plu and span is sing
      ptop_antecedent_scores = tf.concat([dummy_scores, ptop_antecedent_scores], 1)  # [k, c + 1, clu]
      ptop_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, ptop_antecedents)  # [k, c, clu]
      ant_singular = tf.count_nonzero(ptop_antecedent_cluster_ids, 2) <= 1 # [k, c]
      ptop_antecedents_mask = tf.tile(tf.expand_dims(ptop_antecedents_mask, 2), [1, 1, max_clusters])  # [k, c, clu]
      ptop_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(ptop_antecedents_mask)))  # [k, c, clu]
      expanded_span_cluster_ids = tf.tile(tf.expand_dims(top_span_cluster_ids, 1), [1, c, 1])
      same_cluster_indicator = tf.sets.set_intersection(ptop_antecedent_cluster_ids, expanded_span_cluster_ids)
      same_cluster_indicator = sparse2dense(same_cluster_indicator)  # tf.sparse_to_dense()
      same_cluster_indicator = tf.count_nonzero(same_cluster_indicator, 2) > 0  # [k, c]
      # todo plural check for antecedent
      ant_plural = tf.count_nonzero(ptop_antecedent_cluster_ids, 2) >= 2  # [k, c]
      valid_indicator = tf.logical_and(non_dummy_indicator, span_singular if self.ln else span_plural)
      valid_indicator = tf.logical_and(valid_indicator, ant_plural if self.ln else ant_singular) # avoid plural-plural
      pairwise_labels = tf.logical_and(same_cluster_indicator, valid_indicator)  # [k, c]
      dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
      ptop_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]
      ploss = self.softmax_loss(ptop_antecedent_scores, ptop_antecedent_labels)  # [k]
      ploss = tf.reduce_sum(ploss)  # []

      loss = self.singularity * loss + (1-self.singularity) * ploss

    else: # todo clusters
      top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask))) # [k, c]
      same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1)) # [k, c]
      non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1) # [k, 1]
      pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator) # [k, c]
      dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True)) # [k, 1]
      top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1) # [k, c + 1]
      loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels) # [k]
      loss = tf.reduce_sum(loss) # []


    #candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_score
    ptop_antecedents, ptop_antecedent_scores = (ptop_antecedents, ptop_antecedent_scores) if self.plural_clusters else (top_antecedents, top_antecedent_scores)

    return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores,
            ptop_antecedents, ptop_antecedent_scores, top_span_emb], loss

  def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
    span_emb_list = []

    span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]
    span_emb_list.append(span_start_emb)

    span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]
    span_emb_list.append(span_end_emb)

    span_width = 1 + span_ends - span_starts # [k]

    if self.config["use_features"]:
      span_width_index = span_width - 1 # [k]
      span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]), span_width_index) # [k, emb]
      span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
      span_emb_list.append(span_width_emb)

    if self.config["model_heads"]:
      span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts, 1) # [k, max_span_width]
      span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices) # [k, max_span_width]
      span_text_emb = tf.gather(head_emb, span_indices) # [k, max_span_width, emb]
      with tf.variable_scope("head_scores"):
        self.head_scores = util.projection(context_outputs, 1) # [num_words, 1]
      span_head_scores = tf.gather(self.head_scores, span_indices) # [k, max_span_width, 1]
      span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32), 2) # [k, max_span_width, 1]
      span_head_scores += tf.log(span_mask) # [k, max_span_width, 1]
      span_attention = tf.nn.softmax(span_head_scores, 1) # [k, max_span_width, 1]
      span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1) # [k, emb]
      span_emb_list.append(span_head_emb)

    span_emb = tf.concat(span_emb_list, 1) # [k, emb]
    return span_emb # [k, emb]

  def get_mention_scores(self, span_emb):
    with tf.variable_scope("mention_scores"):
      return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, 1]

  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [k, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [k]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [k]
    return log_norm - marginalized_gold_scores # [k]

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)

  def average_speakers(self, x, k, c, s):
    nz = tf.count_nonzero(x, 2)  # [k, c]
    x = tf.gather(tf.get_variable("speaker_emb", [len(self.spk_dict) + 2,
                                                  self.config["feature_size"]]), x)  # [k, c, s, emb]
    nz = tf.to_float(tf.tile(tf.expand_dims(nz, 2), [1, 1, self.config["feature_size"]]))  # [k, c, emb]
    zval = tf.gather(tf.get_variable("speaker_emb"), 0)
    zval = tf.tile(tf.expand_dims(tf.expand_dims(zval, 0), 0), [k, c, 1])  # [k, c, emb]
    sub = (tf.to_float(s) - nz) * zval  # [k, c, emb]
    ssum = tf.reduce_sum(x, 2)  # [k, c, emb]
    return (ssum - sub) / nz  # [k, c, emb]

  # todo plural
  def get_plural_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, genre_emb):
    k = util.shape(top_span_emb, 0)
    c = util.shape(top_antecedents, 1)

    feature_emb_list = []

    if self.config["use_metadata"]:
      top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents) # [k, c]
      if self.spk:
        # top_antecedent_speaker_ids = tf.gather(top_antecedent_speaker_ids, tf.squeeze(tf.where(where), 3  )) # got rid of placeholder zeros
        s = util.shape(top_span_speaker_ids, 1)
        with tf.control_dependencies([tf.assert_rank(top_antecedent_speaker_ids, 3)]):
          top_antecedent_speaker_ids = self.average_speakers(top_antecedent_speaker_ids, k, c, s)
          feature_emb_list.append(top_antecedent_speaker_ids)
        span_spk = tf.tile(tf.expand_dims(top_span_speaker_ids, 1), [1, c, 1]) # [k, c, s]
        with tf.control_dependencies([tf.assert_rank(span_spk, 3)]):
          span_spk = self.average_speakers(span_spk, k, c, s)
          feature_emb_list.append(span_spk)
        feature_emb_list.append(span_spk * top_antecedent_speaker_ids) # [k, c, emb]
      else:
        if self.config["plural_speakers"]:
          s = util.shape(top_span_speaker_ids, 1)
          expanded = tf.reshape(tf.tile(top_span_speaker_ids, [1, c]), [k, c, s])
          same = tf.sets.set_intersection(expanded, top_antecedent_speaker_ids)
          dense = tf.sparse_to_dense(same.indices, same.dense_shape, same.values)
          same_speaker = tf.count_nonzero(dense, 2) > 0 # [k, c]
          #print('same_speaker',same_speaker)
        else:
          same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids) # [k, c]
        speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]), tf.to_int32(same_speaker)) # [k, c, emb]
        feature_emb_list.append(speaker_pair_emb)
        #print('speaker_pair_emb', speaker_pair_emb)
      if self.config["use_genre"]:
        tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1]) # [k, c, emb]
        feature_emb_list.append(tiled_genre_emb)

    if self.config["use_features"]:
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [k, c]
      antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]), antecedent_distance_buckets) # [k, c]
      feature_emb_list.append(antecedent_distance_emb)
      #print('antecedent_distance_emb', antecedent_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [k, c, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [k, c, emb]

    target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
    target_emb = tf.tile(target_emb, [1, c, 1]) # [k, c, emb]

    pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb]

    with tf.variable_scope("plural_slow_antecedent_scores"):
      #print('pair_emb', pair_emb, flush=True)
      slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, c, 1]
    slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2) # [k, c]
    #exit(1)
    return slow_antecedent_scores # [k, c]

  def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, genre_emb):
    k = util.shape(top_span_emb, 0)
    c = util.shape(top_antecedents, 1)

    feature_emb_list = []

    if self.config["use_metadata"]:
      top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents) # [k, c]
      if self.spk:
        s = util.shape(top_span_speaker_ids, 1)
        with tf.control_dependencies([tf.assert_rank(top_antecedent_speaker_ids, 3)]):
          top_antecedent_speaker_ids = self.average_speakers(top_antecedent_speaker_ids, k, c, s)
          feature_emb_list.append(top_antecedent_speaker_ids)
        span_spk = tf.tile(tf.expand_dims(top_span_speaker_ids, 1), [1, c, 1])  # [k, c, s]
        with tf.control_dependencies([tf.assert_rank(span_spk, 3)]):
          span_spk = self.average_speakers(span_spk, k, c, s)
          feature_emb_list.append(span_spk)
        feature_emb_list.append(span_spk * top_antecedent_speaker_ids)  # [k, c, emb]
      else:
        if self.config["plural_speakers"]:
          s = util.shape(top_span_speaker_ids, 1)
          expanded = tf.reshape(tf.tile(top_span_speaker_ids, [1, c]), [k, c, s])
          same = tf.sets.set_intersection(expanded, top_antecedent_speaker_ids)
          dense = tf.sparse_to_dense(same.indices, same.dense_shape, same.values)
          same_speaker = tf.count_nonzero(dense, 2) > 0 # [k, c]
        else:
          same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids) # [k, c]
        speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]), tf.to_int32(same_speaker)) # [k, c, emb]
        feature_emb_list.append(speaker_pair_emb)
      if self.config["use_genre"]:
        tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1]) # [k, c, emb]
        feature_emb_list.append(tiled_genre_emb)

    if self.config["use_features"]:
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [k, c]
      antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]), antecedent_distance_buckets) # [k, c]
      feature_emb_list.append(antecedent_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [k, c, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [k, c, emb]

    target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
    target_emb = tf.tile(target_emb, [1, c, 1]) # [k, c, emb]

    pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb]

    with tf.variable_scope("slow_antecedent_scores"):
      slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, c, 1]
    slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2) # [k, c]
    return slow_antecedent_scores # [k, c]


  # todo plural
  def get_plural_fast_antecedent_scores(self, top_span_emb):
    with tf.variable_scope("plural_src_projection"):
      source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)), self.dropout) # [k, emb]
    target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout) # [k, emb]
    return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [k, k]


  def get_fast_antecedent_scores(self, top_span_emb):
    with tf.variable_scope("src_projection"):
      source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)), self.dropout) # [k, emb]
    target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout) # [k, emb]
    return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [k, k]

  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank  == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

  def lstm_contextualize(self, text_emb, text_len, text_len_mask):
    num_sentences = tf.shape(text_emb)[0]

    current_inputs = text_emb # [num_sentences, max_sentence_length, emb]

    for layer in range(self.config["contextualization_layers"]):
      with tf.variable_scope("layer_{}".format(layer)):
        with tf.variable_scope("fw_cell"):
          cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        with tf.variable_scope("bw_cell"):
          cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]), tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
        state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]), tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

        (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=current_inputs,
          sequence_length=text_len,
          initial_state_fw=state_fw,
          initial_state_bw=state_bw)

        text_outputs = tf.concat([fw_outputs, bw_outputs], 2) # [num_sentences, max_sentence_length, emb]
        text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
        if layer > 0:
          highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs, 2))) # [num_sentences, max_sentence_length, emb]
          text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
        current_inputs = text_outputs

    return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

  def get_predicted_antecedents(self, antecedents, antecedent_scores):
    if self.debug:
      print('--Original_predicted')
      #print('antecedents', antecedents.shape, antecedents)
      #print('antecedent_scores', antecedent_scores.shape, antecedent_scores)
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
      if index < 0:
        predicted_antecedents.append(-1)  # thresh = np.argmax
      else:
        predicted_antecedents.append(antecedents[i, index])
    #print('predicted_antecedents',len(predicted_antecedents), predicted_antecedents)
    if self.debug:
      print('pred ants:', [(i,x) for i,x in enumerate(predicted_antecedents) if x >= 0])
    return predicted_antecedents

  def get_singular_predicted_antecedents(self, antecedents, antecedent_scores):
    if self.debug:
      print('--Singular_predicted')
      #print('SINGULAR antecedents', antecedents.shape, antecedents)
      #print('antecedent_scores', antecedent_scores.shape, antecedent_scores)
    ans = []
    for r, row in enumerate(antecedent_scores):
      thresh = row[0]
      predicted_antecedents = []
      for score, i in sorted([(s, i) for i, s in enumerate(row)])[-self.L:]:
      #for i, score in enumerate(row):
        if score > thresh:
          predicted_antecedents.append(antecedents[r, i-1])  # thresh = dummy
      ans.append(predicted_antecedents)
    if self.debug:
      print('pred ant', [(i,x) for i,x in enumerate(ans) if x])
    return ans

  def get_plural_predicted_antecedents(self, antecedents, antecedent_scores):
    if self.debug:
      print('--Plural_predicted')
      #print('PLURAL antecedents', antecedents.shape, antecedents)
      #print('antecedent_scores', antecedent_scores.shape, antecedent_scores)
    ans = []
    for r, row in enumerate(antecedent_scores):
      thresh = row[0]
      predicted_antecedents = []
      #for i, score in enumerate(row):
      for score, i in sorted([(s, i) for i, s in enumerate(row)])[-self.R:]:
        if score > thresh:
          predicted_antecedents.append(antecedents[r, i-1])  # thresh = dummy
      ans.append(predicted_antecedents)
    if self.debug:
      print('pred ant', [(i,x) for i,x in enumerate(ans) if x])
    return ans

  def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      # Change > to >= since  == for singletons
      assert i > predicted_index
      predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(top_span_starts[i]), int(top_span_ends[i]))
      #if self.debug: print('join span {} with ant {} at cluster {}'.format(mention, predicted_antecedent, predicted_cluster))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    #mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }
    #if self.debug and self.plural_clusters:
      #print('top_span_starts', top_span_starts.shape, top_span_starts)
      #print('predicted_antecedents', len(predicted_antecedents), predicted_antecedents)
      #print('predicted_clusters', len(predicted_clusters), predicted_clusters)
      #print('mention_to_predicted', len(mention_to_predicted), mention_to_predicted)
    return predicted_clusters, mention_to_predicted

  # Add each span to the cluster of the ant
  def get_singular_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
    #if self.debug: print('ant starts', [[top_span_starts[ant] for ant in span] for span in predicted_antecedents])
    mention_to_predicted = {}
    predicted_clusters = []
    for i,row in enumerate(predicted_antecedents):
      for predicted_index in row:
        assert i > predicted_index
        mention = (int(top_span_starts[i]), int(top_span_ends[i]))
        predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
        # """
        if predicted_antecedent in mention_to_predicted:
          predicted_cluster = mention_to_predicted[predicted_antecedent]
        else:
          predicted_cluster = len(predicted_clusters)
          predicted_clusters.append([predicted_antecedent])
          mention_to_predicted[predicted_antecedent] = predicted_cluster
        predicted_clusters[predicted_cluster].append(mention)
        mention_to_predicted[mention] = predicted_cluster
        """
        if mention in mention_to_predicted:
          predicted_cluster = mention_to_predicted[mention]
        else:
          predicted_cluster = len(predicted_clusters)
          predicted_clusters.append([mention])
          mention_to_predicted[mention] = predicted_cluster
        predicted_clusters[predicted_cluster].append(predicted_antecedent)
        mention_to_predicted[predicted_antecedent] = predicted_cluster
        """

        predicted_clusters[predicted_cluster].append(predicted_antecedent)
        mention_to_predicted[predicted_antecedent] = predicted_cluster
        #if self.debug: print('join span {} with ant {} at cluster {}'.format(mention, predicted_antecedent, predicted_cluster))
    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    #if self.debug: print('sing pred clsuters', predicted_clusters)
    return predicted_clusters, mention_to_predicted

  def get_comb_predicted_antecedents(self, singular, singular_scores, plural, plural_scores):
    if self.debug:
      print('--comb_predicted_ants')
      #print('SINGULAR antecedents', singular.shape, singular)
      #print('PLURAL antecedents', plural.shape, plural)
      #print('antecedent_scores', antecedent_scores.shape, antecedent_scores)
    ans = []
    for r, (srow, prow) in enumerate(zip(singular_scores, plural_scores)):
      st, pt = srow[0], prow[0]
      predicted_antecedents = {}
      for score, i in sorted([(s,i) for i,s in enumerate(srow)])[-self.L:]:
      #for i, score in enumerate(srow):
        if score > st:
          predicted_antecedents[singular[r, i-1]] = (score, (singular[r, i-1], False))  # thresh = dummy
      #"""
      for score, i in sorted([(s, i) for i, s in enumerate(prow)])[-self.R:]:
        if score > pt and score > predicted_antecedents.get(plural[r, i-1], (pt, None))[0]:
          predicted_antecedents[plural[r, i-1]] = (score, (plural[r, i-1], True)) # true for plural
      #"""
      ans.append(sorted([x[1] for x in predicted_antecedents.values()])) # dont need score
    if self.debug: print('pred ants', [(i,x) for i,x in enumerate(ans) if x])
    return ans


  # ant < i => ant=mi and span=mj
  def get_comb_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
    flat = sorted([(isp, i, ant) for i, span in enumerate(predicted_antecedents) for ant, isp in span if span])
    if self.debug:
      print('-comb clusters')
      print('pants', predicted_antecedents)
      print('flat', flat)
      #print('ants', predicted_antecedents)
      #print('ant starts', [(top_span_starts[i],[(top_span_starts[ant],isp) for ant, isp in span]) for i,span in enumerate(predicted_antecedents) if span])
    mention_to_predicted = {}
    predicted_clusters = []
    assert(len(predicted_antecedents) == top_span_starts.shape[0])

    #if self.debug:print('flat', len(flat), flat)
    #predicted_antecedents = [(i,ant,False) for i,ant in [(3,1), (4,1), (4,3), (7,0), (7,6), (9,1), (9,3), (9,4), (10,6), (12,6), (12,10), (13,11)]]
    #predicted_antecedents += [(8,4,True)]
    #for i,row in enumerate(predicted_antecedents):
    for is_plural, i, predicted_index in flat:
      #for predicted_index, is_plural in row:
        assert i > predicted_index
        mention = (int(top_span_starts[i]), int(top_span_ends[i]))
        predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
        if is_plural:
          if predicted_antecedent in mention_to_predicted:
            predicted_cluster = mention_to_predicted[predicted_antecedent]
          else:
            predicted_cluster = len(predicted_clusters)
            predicted_clusters.append([predicted_antecedent])
            mention_to_predicted[predicted_antecedent] = predicted_cluster
          predicted_clusters[predicted_cluster].append(mention)
          mention_to_predicted[mention] = predicted_cluster
        else: # is_singular
          if mention in mention_to_predicted:
            predicted_cluster = mention_to_predicted[mention]
          else:
            predicted_cluster = len(predicted_clusters)
            predicted_clusters.append([mention])
            mention_to_predicted[mention] = predicted_cluster
          predicted_clusters[predicted_cluster].append(predicted_antecedent)
          mention_to_predicted[predicted_antecedent] = predicted_cluster
    #predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    predicted_clusters = [tuple(set(tuple(pc))) for pc in predicted_clusters]
    return predicted_clusters, mention_to_predicted

    # ant < i => ant=mi and span=mj
  def get_ln_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
      #predicted_antecedents = sorted([(mj, mi, isp) for mj,span in enumerate(predicted_antecedents) if span for mi,isp in span])
      plural_ants = sorted(
        [(mj, mi) for mj, span in enumerate(predicted_antecedents) if span for mi, isp in span if isp])
      sing_ants = sorted(
        [(mj, mi) for mj, span in enumerate(predicted_antecedents) if span for mi, isp in span if not isp])
      #sing_ants = [(3,1), (4,1), (4,3), (7,0), (7,6), (9,1), (9,3), (9,4), (10,6), (12,6), (12,10), (13,11)]
      #plural_ants = [(8,4)]
      mention_to_predicted = {}
      predicted_clusters = []
      for mj, mi in sing_ants:
        assert mj > mi
        mj, mi = (int(top_span_starts[mj]), int(top_span_ends[mj])), (int(top_span_starts[mi]), int(top_span_ends[mi]))
        if mi in mention_to_predicted:
          predicted_cluster = mention_to_predicted[mi]
        else:
          predicted_cluster = len(predicted_clusters)
          predicted_clusters.append([mi])
          mention_to_predicted[mi] = predicted_cluster
        predicted_clusters[predicted_cluster].append(mj)
        mention_to_predicted[mj] = predicted_cluster
      for mj, mi in plural_ants:
        assert mj > mi
        #if mi in m2i and mj in m2i and uf.connected(m2i[mj], m2i[mi]): continue
        mj, mi = (int(top_span_starts[mj]), int(top_span_ends[mj])), (int(top_span_starts[mi]), int(top_span_ends[mi]))
        if self.debug:
          print('plural {} with sing {}'.format(mi,mj))
          self.pwords.append(mi)
        if mj in mention_to_predicted:
          predicted_cluster = mention_to_predicted[mj]
        else:
          predicted_cluster = len(predicted_clusters)
          predicted_clusters.append([mj])
          mention_to_predicted[mj] = predicted_cluster
        predicted_clusters[predicted_cluster].append(mi)
        mention_to_predicted[mi] = predicted_cluster
      #if self.debug: print('after plur', [(set(tuple(pc))) for pc in predicted_clusters])
      predicted_clusters = [tuple(set(tuple(pc))) for pc in predicted_clusters]
      return predicted_clusters, mention_to_predicted

  def get_lncomb_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
      #ants = sorted([(mj, mi, isp) for mj, span in enumerate(predicted_antecedents) if span for mi, isp in span])
      ants = sorted([(isp, mj, mi) for mj, span in enumerate(predicted_antecedents) if span for mi, isp in span])
      mention_to_predicted = {}
      predicted_clusters = []
      #for mj, mi, is_plural in ants:
      for is_plural, mj, mi in ants:
        assert mj > mi
        mj, mi = (int(top_span_starts[mj]), int(top_span_ends[mj])), (int(top_span_starts[mi]), int(top_span_ends[mi]))
        if is_plural:
          if mj in mention_to_predicted:
            predicted_cluster = mention_to_predicted[mj]
          else:
            predicted_cluster = len(predicted_clusters)
            predicted_clusters.append([mj])
            mention_to_predicted[mj] = predicted_cluster
          predicted_clusters[predicted_cluster].append(mi)
          mention_to_predicted[mi] = predicted_cluster
        else:
          if mi in mention_to_predicted:
            predicted_cluster = mention_to_predicted[mi]
          else:
            predicted_cluster = len(predicted_clusters)
            predicted_clusters.append([mi])
            mention_to_predicted[mi] = predicted_cluster
          predicted_clusters[predicted_cluster].append(mj)
          mention_to_predicted[mj] = predicted_cluster
      predicted_clusters = [tuple(set(tuple(pc))) for pc in predicted_clusters]
      return predicted_clusters, mention_to_predicted

  # Add each ant to the cluster of the span
  def get_plural_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents, mention_to_predicted, singular_clusters):
    predicted_clusters = [list(c) for c in singular_clusters]
    if self.debug:
      print('--Plural pred clust')
      print('ant starts', [(top_span_starts[i],[top_span_starts[ant] for ant in span]) for i,span in enumerate(predicted_antecedents) if span])

      #print('singular_clusters',len(singular_clusters), singular_clusters)
      #print('predicted_antecedents', len(predicted_antecedents), predicted_antecedents)

    if not self.config["pluants"]:
      predicted_antecedents = [[x] if x>= 0 else [] for x in predicted_antecedents]
      #predicted_antecedents = [ant for i,span in enumerate(predicted_antecedents) for ant in span]

    #for i, predicted_index in enumerate(predicted_antecedents):
    for i,row in enumerate(predicted_antecedents):
      for predicted_index in row:
        assert i > predicted_index
        predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
        mention = (int(top_span_starts[i]), int(top_span_ends[i]))
        # """
        if predicted_antecedent in mention_to_predicted:
          predicted_cluster = mention_to_predicted[predicted_antecedent]
        else:
          predicted_cluster = len(predicted_clusters)
          predicted_clusters.append([predicted_antecedent])
          mention_to_predicted[predicted_antecedent] = predicted_cluster
        predicted_clusters[predicted_cluster].append(mention)
        mention_to_predicted[mention] = predicted_cluster
        """
        if mention in mention_to_predicted:
          predicted_cluster = mention_to_predicted[mention]
        else:
          predicted_cluster = len(predicted_clusters)
          predicted_clusters.append([mention])
          mention_to_predicted[mention] = predicted_cluster
        predicted_clusters[predicted_cluster].append(predicted_antecedent)
        mention_to_predicted[predicted_antecedent] = predicted_cluster
        """
    predicted_clusters = [tuple(set(tuple(pc))) for pc in predicted_clusters]
    #mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }
    #if self.debug:
    #  print('predicted_clusters', len(predicted_clusters), predicted_clusters)
    #  print('mention_to_predicted', len(mention_to_predicted))#, mention_to_predicted)
    return predicted_clusters, mention_to_predicted



  def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
    gold_clusters = [tuple(sorted(tuple(m) for m in gc)) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc
    if self.debug and self.plural_clusters: print('gold_clusters', len(gold_clusters), gold_clusters)
    predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters

  def load_eval_data(self, test=False):
    if self.eval_data is None:
      def load_line(line):
        example = json.loads(line)
        return self.tensorize_example(example, is_training=False), example
      with open(self.config["eval_path" if not test else "test_path"]) as f:
        self.eval_data = [load_line(l) for l in f.readlines()]
      num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
      print("Loaded {} eval examples.".format(len(self.eval_data)))

  def score(self, golds, autos, show=True):
    if len(golds) != len(set(golds)) or list(len(x) for x in golds) != list(len(set(x)) for x in golds):
    #if tuple(set(tuple(set(x)) for x in golds)) != golds:
      print('repeat golds', golds)
    if len(autos) != len(set(autos)) or list(len(x) for x in autos) != list(len(set(x)) for x in autos):
    #if tuple(set(tuple(set(x)) for x in autos)) != autos:
      print('repeat autos', autos)
    if show: print(len(autos), list(map(len,autos)))
    golds = [golds]
    autos = [autos]
    f_total = 0
    p, r, f = evaluators.MentionEvaluator().evaluate_documents(golds, autos)
    if show: print('Mention Detection - %.4f/%.4f/%.4f' % (p, r, f))
    p, r, f = evaluators.BCubeEvaluator().evaluate_documents(golds, autos)
    f_total += f
    if show: print('Bcube - %.4f/%.4f/%.4f' % (p, r, f))
    p, r, f = evaluators.CeafeEvaluator().evaluate_documents(golds, autos)
    f_total += f
    if show: print('Ceafe - %.4f/%.4f/%.4f' % (p, r, f))
    p, r, f = evaluators.BlancEvaluator().evaluate_documents(golds, autos)
    f_total += f
    if show: print('Blanc - %.4f/%.4f/%.4f' % (p, r, f))
    if show: print('Avg F1 - %.4f' % (f_total/3))
    return f_total/3


  def plot(self, span_emb, starts, ends, clusters, doc="None", score=-1, sentences=None, golds=None):
      i2s = {(starts[i], ends[i]): i for i in range(len(starts))}
      #print('c0', clusters[0])
      #print('cl', clusters)
      #print('emb', span_emb.shape, span_emb)
      old = clusters
      clusters = [[i2s[x] for x in cluster] for cluster in clusters]
      golds = [[i2s[x] for x in cluster if x in i2s] for cluster in golds]
      #simil = [cosine_similarity(np.array([span_emb[i] for i in cluster])) for cluster in clusters if len(cluster) > 1]
      #means = [np.mean(x) for x in simil]
      #avg = sum(means) / len(means)
      arr = np.array([span_emb[i] for c in clusters for i in c])

      ex = "ln" if self.ln else "other"

      tsne = TSNE(n_components=2, random_state=0)
      np.set_printoptions(suppress=True)
      Y = tsne.fit_transform(arr)

      x_coords = Y[:, 0]
      y_coords = Y[:, 1]
      names = [i for i, c in enumerate(clusters) for _ in c]
      gold_names = [i for i, c in enumerate(golds) for _ in c]
      #print('sent', sentences)
      #print('old', old)
      words = ['_'.join(sentences[x:y+1]) for c in old for x,y in c]
      #print('w', words)
      #exit()
      # display scatter plot
      colors = cm.rainbow(np.linspace(0, 1, max(len(clusters), len(golds))))
      for xc, yc, name in zip(x_coords, y_coords, names):
        plt.scatter(xc, yc, color=colors[name])

      #plt.scatter(x_coords, y_coords)

      #for label, x, y in zip(words, x_coords, y_coords): plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
      plt.xlim(x_coords.min() * 1.1, x_coords.max() * 1.1)
      plt.ylim(y_coords.min() * 1.1, y_coords.max() * 1.1)
      plt.savefig('cluster_{}_%.3f_{}.png'.format(ex, doc) % score)

      for xc, yc, name in zip(x_coords, y_coords, gold_names):
        plt.scatter(xc, yc, color=colors[name])
      plt.xlim(x_coords.min() * 1.1, x_coords.max() * 1.1)
      plt.ylim(y_coords.min() * 1.1, y_coords.max() * 1.1)
      plt.savefig('gold_{}_%.3f_{}.png'.format(ex, doc) % score)

      #plt.show()
      #print(arr.shape)
      #print(Y.shape)
      #print('Done')
      #exit()


  def evaluate(self, session, log_dir, official_stdout=False, test=False,):
    self.load_eval_data(test)

    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()
    needed = []
    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      #if self.debug and example["doc_key"] != "s01_e20_c01_0": continue
      gold_clusters = [tuple(tuple(m) for m in gc) for gc in example["clusters"]]
      if self.debug: print('gold',len(gold_clusters), sorted(gold_clusters))
      if self.debug and self.plural_clusters:
        mf = collections.Counter()
        for gc in gold_clusters:
          for m in gc:
            mf[m] += 1
        if all([x <= 1 for x in mf.values()]):
          print('No plural')
          continue
        print('KEY', example["doc_key"], mf)
        print('SINGULAR-----------------------------')
      _, _, _, _, _, _, _, _, _, gold_starts, gold_ends, _ = tensorized_example
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      #candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(self.predictions, feed_dict=feed_dict)
      candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores, \
      ptop_antecedents, ptop_antecedent_scores, top_spam_emb = session.run(self.predictions, feed_dict=feed_dict)

      if self.debug:
        ss = [w for s in example["sentences"] for w in s]
        print(ss)
        sing = self.get_singular_predicted_antecedents(top_antecedents, top_antecedent_scores)
        original = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
        sing_clusters, m2p = self.get_singular_predicted_clusters(top_span_starts, top_span_ends, sing)
        ori_clusters, _ = self.get_predicted_clusters(top_span_starts, top_span_ends, original)
        print('sing clusters', sorted(sing_clusters))
        #self.score(gold_clusters, sing_clusters)
        print('ori clusters', sorted(sing_clusters))
        self.score(gold_clusters, ori_clusters)

        # plu = self.get_plural_predicted_antecedents(ptop_antecedents, ptop_antecedent_scores)
        #plural_clusters, _ = self.get_plural_predicted_clusters(top_span_starts, top_span_ends, plu, m2p, sing_clusters)
        #print('plu clsuters', sorted(sorted(x) for x in plural_clusters))
        #self.score(gold_clusters, plural_clusters)

        self.pwords = []
        comb_ants = self.get_comb_predicted_antecedents(top_antecedents, top_antecedent_scores, ptop_antecedents, ptop_antecedent_scores)

        comb_clusters, _ = self.get_comb_predicted_clusters(top_span_starts, top_span_ends, comb_ants)
        print('comb clusters', sorted(sorted(x) for x in comb_clusters))
        self.score(gold_clusters, comb_clusters)

        ln_clusters, _ = self.get_ln_predicted_clusters(top_span_starts, top_span_ends, comb_ants)
        print('ln clusters',  sorted(sorted(x) for x in ln_clusters))
        self.score(gold_clusters, ln_clusters)
        self.plot(top_spam_emb, top_span_starts, top_span_ends, ln_clusters)

        print('plural words', [ss[x:y + 1] for x, y in self.pwords])

        print('gold clusters', sorted(sorted(x) for x in gold_clusters))
        self.score(gold_clusters, gold_clusters)
        exit()

      if self.comb:
        results, _ = self.get_comb_predicted_clusters(top_span_starts,top_span_ends, self.get_comb_predicted_antecedents(
          top_antecedents, top_antecedent_scores, ptop_antecedents, ptop_antecedent_scores))
      elif self.config["lncomb"]:
        results, _ = self.get_lncomb_predicted_clusters(top_span_starts, top_span_ends, self.get_comb_predicted_antecedents(
          top_antecedents, top_antecedent_scores, ptop_antecedents, ptop_antecedent_scores))
      elif self.ln:
        results, _ = self.get_ln_predicted_clusters(top_span_starts, top_span_ends, self.get_comb_predicted_antecedents(
          top_antecedents, top_antecedent_scores, ptop_antecedents, ptop_antecedent_scores))
      else:
        predicted_antecedents = self.get_singular_predicted_antecedents(top_antecedents, top_antecedent_scores) if self.many_antecedents \
            else self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
        results, mention_predicted = self.get_singular_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents) \
            if self.many_antecedents else self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
        if self.plural_clusters:
          if self.debug: print('PLURAL---------------------------------')
          plural_antecedents = self.get_plural_predicted_antecedents(ptop_antecedents, ptop_antecedent_scores) if self.config["pluants"] \
            else self.get_predicted_antecedents(ptop_antecedents, ptop_antecedent_scores)
          results, _ = self.get_plural_predicted_clusters(top_span_starts, top_span_ends, plural_antecedents, mention_predicted, results)

      if self.config["plot"] and results:
        self.plot(top_spam_emb, top_span_starts, top_span_ends, results, example["doc_key"],
                  self.score([tuple(tuple(m) for m in gc) for gc in example["clusters"]], results, show=False),
                  [w for s in example["sentences"] for w in s], [tuple(tuple(m) for m in gc) for gc in example["clusters"]])

      seen = set(x for sublist in results for x in sublist)
      need = list(((b,e),) for b,e,x in zip(candidate_starts, candidate_starts, candidate_mention_scores) if x>self.singleton_thresh and (b,e) not in seen)
      results = [tuple(set(cluster)) for cluster in results]
      needed += [len(need)]
      if len(need) > 0:
        if self.debug:
          print('Need extra', len(need), example["doc_key"])
          print('need', need)
        results += (*need,)



      coref_predictions[example["doc_key"]] = results
      #print('coref_predictions',coref_predictions)
      if self.debug:
        predicted_clusters = results
        ss = example["sentences"]
        print('sentences', ss)
        print('predicted_clusters', len(predicted_clusters), predicted_clusters)
        print('gold', len(gold_clusters), gold_clusters)
        ss = [w for s in ss for w in s]
        print('pred', [[(ss[x:y+1]) for x, y in cluster] for cluster in predicted_clusters])
        print('gold', [[(ss[x:y+1]) for x, y in cluster] for cluster in gold_clusters])
        exit(1)

      if example_num % 10 == 0:
        print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))
    if sum(needed) > 0: print('Needed',sum(needed), needed)
    summary_dict = {}
    conll_results = conll.evaluate_conll(self.config["conll_eval_path" if not test else "conll_test_path"], coref_predictions, log_dir, official_stdout)
    average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    summary_dict["Average F1 (conll)"] = average_f1
    print("Average F1 (conll): {:.3f}%".format(average_f1))

    p,r,f = coref_evaluator.get_prf()
    summary_dict["Average F1 (py)"] = f
    #print("Average F1 (py): {:.2f}%".format(f * 100))
    summary_dict["Average precision (py)"] = p
    #print("Average precision (py): {:.2f}%".format(p * 100))
    summary_dict["Average recall (py)"] = r
    #print("Average recall (py): {:.2f}%".format(r * 100))

    return util.make_summary(summary_dict), average_f1
