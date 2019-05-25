from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import json
import tempfile
import subprocess
import collections

import util
import conll

Names = []
Name2id = {}

def getNameId(name, language):
  #if language != 'friends':
  try:
    return int(name)
  except:
    if name not in Name2id:
      Name2id[name] = len(Names)
      Names.append(name)
    return Name2id[name]

class DocumentState(object):
  def __init__(self):
    self.doc_key = None
    self.text = []
    self.text_speakers = []
    self.speakers = []
    self.sentences = []
    self.constituents = {}
    self.const_stack = []
    self.ner = {}
    self.ner_stack = []
    self.clusters = collections.defaultdict(set)
    self.coref_stacks = collections.defaultdict(list)

  def assert_empty(self):
    assert self.doc_key is None
    assert len(self.text) == 0
    assert len(self.text_speakers) == 0
    assert len(self.speakers) == 0
    assert len(self.sentences) == 0
    assert len(self.constituents) == 0
    assert len(self.const_stack) == 0
    assert len(self.ner) == 0
    assert len(self.ner_stack) == 0
    assert len(self.coref_stacks) == 0
    assert len(self.clusters) == 0

  def assert_finalizable(self):
    assert self.doc_key is not None
    assert len(self.text) == 0
    assert len(self.text_speakers) == 0
    assert len(self.speakers) > 0
    assert len(self.sentences) > 0
    assert len(self.constituents) > 0
    assert len(self.const_stack) == 0
    assert len(self.ner_stack) == 0
    assert all(len(s) == 0 for s in self.coref_stacks.values())

  def span_dict_to_list(self, span_dict):
    return [(s,e,l) for (s,e),l in span_dict.items()]

  def finalize(self):
    merges = 0
    merged_clusters = []
    #print('self.clusters.values()',self.clusters.values())
    #exit(1)
    #print('sent', len(self.sentences),self.sentences)
    #print('self.clusters', len(self.clusters), self.clusters)
    #print('self.clusters.values()', self.clusters.values())
    #exit(1)
    """
    for c1 in self.clusters.values():
      existing = None
      for m in c1:
        for c2 in merged_clusters:
          if m in c2:
            existing = c2
            break
        if existing is not None:
          break
      if existing is not None:
        #print("Merging clusters (shouldn't happen very often.)")
        merges += 1
        existing.update(c1)
      else:
        merged_clusters.append(set(c1))
    if merges>0: print("Merging clusters (shouldn't happen very often.) =", merges)
    merged_clusters = [list(c) for c in merged_clusters]
    all_mentions = util.flatten(merged_clusters)
    print('merged_clusters', len(merged_clusters), merged_clusters)
    ans = [list(c) for c in self.clusters.values()]
    print('ans,', len(ans), ans)
    exit(1)

    if not len(all_mentions) == len(set(all_mentions)):
      ss = sorted(all_mentions)
      for i in range(len(ss) - 1):
        if ss[i] == ss[i+1]:
          print(ss[i])
      print(ss)
    assert len(all_mentions) <= len(set(all_mentions)) + 1
    """
    # assert len(all_mentions) == len(set(all_mentions))
    # print(len(all_mentions), len(set(all_mentions)))
    #print('self.clusters.values()', list(self.clusters.values()))
    clusters = set(tuple(set(tuple(c))) for c in self.clusters.values())
    #print('clusters', clusters)
    old = {
      "doc_key": self.doc_key,
      "sentences": self.sentences,
      "speakers": self.speakers,
      #"constituents": self.span_dict_to_list(self.constituents),
      #"ner": self.span_dict_to_list(self.ner),
      "clusters": [list(c) for c in clusters] # merged_clusters
    }
    #print('old', old["clusters"])
    #exit(1)
    ans = collections.OrderedDict()
    for k,v in old.items():
      ans[k] = v
    return ans

def normalize_word(word, language):
  if language == "arabic":
    word = word[:word.find("#")]
  if word == "/." or word == "/?":
    return word[1:]
  else:
    return word

def handle_bit(word_index, bit, stack, spans):
  asterisk_idx = bit.find("*")
  if asterisk_idx >= 0:
    open_parens = bit[:asterisk_idx]
    close_parens = bit[asterisk_idx + 1:]
  else:
    open_parens = bit[:-1]
    close_parens = bit[-1]

  current_idx = open_parens.find("(")
  while current_idx >= 0:
    next_idx = open_parens.find("(", current_idx + 1)
    if next_idx >= 0:
      label = open_parens[current_idx + 1:next_idx]
    else:
      label = open_parens[current_idx + 1:]
    stack.append((word_index, label))
    current_idx = next_idx

  for c in close_parens:
    if c!=')': print(c,'instead of )')
    assert c == ")"
    open_index, label = stack.pop()
    current_span = (open_index, word_index)
    """
    if current_span in spans:
      spans[current_span] += "_" + label
    else:
      spans[current_span] = label
    """
    spans[current_span] = label

def handle_line(line, document_state, language, labels, stats):
  #print(line)
  begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
  if begin_document_match:
    document_state.assert_empty()
    document_state.doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
    return None
  elif line.startswith("#end document"):
    document_state.assert_finalizable()
    finalized_state = document_state.finalize()
    stats["num_clusters"] += len(finalized_state["clusters"])
    stats["num_mentions"] += sum(len(c) for c in finalized_state["clusters"])
    #labels["{}_const_labels".format(language)].update(l for _, _, l in finalized_state["constituents"])
    #labels["ner"].update(l for _, _, l in finalized_state["ner"])
    return finalized_state
  else:
    #row = line.split()
    row = line.split('\t') if language == 'friends' else line.split()
    #print(row)
    if len(row) == 0 or row[0] == '\n':
      stats["max_sent_len_{}".format(language)] = max(len(document_state.text), stats["max_sent_len_{}".format(language)])
      stats["num_sents_{}".format(language)] += 1
      document_state.sentences.append(tuple(document_state.text))
      del document_state.text[:]
      document_state.speakers.append(tuple(document_state.text_speakers))
      del document_state.text_speakers[:]
      return None
    assert len(row) >= 12

    doc_key = conll.get_doc_key(row[0], row[1])
    word = normalize_word(row[3], language)
    parse = row[5]
    speaker = row[9]
    #speakers = row[9].split('|') # todo plural speakers
    ner = row[10]
    coref = row[-1].rstrip()  #row[-1][:-1] if language=='friends' else row[-1]

    word_index = len(document_state.text) + sum(len(s) for s in document_state.sentences)
    document_state.text.append(word)
    document_state.text_speakers.append(speaker)

    handle_bit(word_index, parse, document_state.const_stack, document_state.constituents)
    handle_bit(word_index, ner, document_state.ner_stack, document_state.ner)

    if coref != "-":

      for segment in coref.split("|"):
        #if language=='friends' and word_index==3032: print(row, segment)
        if segment[0] == "(":
          if segment[-1] == ")":
            cluster_id = getNameId(segment[1:-1], language)
            document_state.clusters[cluster_id].add((word_index, word_index))
          else:
            cluster_id = getNameId(segment[1:], language)
            document_state.coref_stacks[cluster_id].append(word_index)
        else:
          cluster_id = getNameId(segment[:-1], language)
          start = document_state.coref_stacks[cluster_id].pop()
          document_state.clusters[cluster_id].add((start, word_index))
    return None

def minimize_partition(name, language, extension, labels, stats):
  input_path = "{}.{}.{}".format(name, language, extension)
  output_path = "{}.{}.jsonlines".format(name, language)
  count = 0
  print("Minimizing {}".format(input_path))
  with open(input_path, "r") as input_file:
    with open(output_path, "w") as output_file:
      document_state = DocumentState()
      for line in input_file.readlines():
        document = handle_line(line, document_state, language, labels, stats)
        if document is not None:
          output_file.write(json.dumps(document))
          output_file.write("\n")
          count += 1
          document_state = DocumentState()
  print("Wrote {} documents to {}".format(count, output_path))

def minimize_language(language, labels, stats):
  minimize_partition("dev", language, "v4_gold_conll", labels, stats)
  minimize_partition("train", language, "v4_gold_conll", labels, stats)
  #if language == 'friends':
    #minimize_partition("singMin_train", language, "v4_gold_conll", labels, stats)
    #minimize_partition("singMax_train", language, "v4_gold_conll", labels, stats)
  minimize_partition("test", language, "v4_gold_conll", labels, stats)

if __name__ == "__main__":
  labels = collections.defaultdict(set)
  stats = collections.defaultdict(int)
  #minimize_language("english", labels, stats)
  #exit(1)
  minimize_language('friends', labels, stats)
  #minimize_language("chinese", labels, stats)
  #minimize_language("arabic", labels, stats)
  for k, v in labels.items():
    print("{} = [{}]".format(k, ", ".join("\"{}\"".format(label) for label in v)))
  for k, v in stats.items():
    print("{} = {}".format(k, v))
  minimize_partition("sing_top_train", 'friends', "v4_gold_conll", labels, stats)
  minimize_partition("sing_min_train", 'friends', "v4_gold_conll", labels, stats)
  minimize_partition("sing_max_train", 'friends', "v4_gold_conll", labels, stats)
  minimize_partition("sing_none_train", 'friends', "v4_gold_conll", labels, stats)
