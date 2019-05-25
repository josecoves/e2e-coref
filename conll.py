from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import json
import tempfile
import subprocess
import operator
import collections
from evaluators import *

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")
COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)

def get_doc_key(doc_id, part):
  return "{}_{}".format(doc_id, int(part))

def output_conll(input_file, output_file, predictions):
  prediction_map = {}
  for doc_key, clusters in predictions.items():
    start_map = collections.defaultdict(list)
    end_map = collections.defaultdict(list)
    word_map = collections.defaultdict(list)
    for cluster_id, mentions in enumerate(clusters):
      for start, end in mentions:
        if start == end:
          word_map[start].append(cluster_id)
        else:
          start_map[start].append((cluster_id, end))
          end_map[end].append((cluster_id, start))
    for k,v in start_map.items():
      start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
    for k,v in end_map.items():
      end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
    prediction_map[doc_key] = (start_map, end_map, word_map)

  word_index = 0
  for line in input_file.readlines():
    row = line.split()
    #print('row', row)
    #if len(row) > 0:print('row[0][0]', row[0][0], row[0][0]=='s', line.split('\t'))
    if len(row) > 0 and row[0][0] == 's':  row = line.split('\t')
    if len(row) == 0 or row[0]=='\n':
      output_file.write("\n")
    elif row[0].startswith("#"):
      begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
      if begin_match:
        doc_key = get_doc_key(begin_match.group(1), begin_match.group(2))
        start_map, end_map, word_map = prediction_map[doc_key]
        word_index = 0
      output_file.write(line)
      output_file.write("\n")
    else:
      #assert get_doc_key(row[0], row[1]) == doc_key
      coref_list = []
      if word_index in end_map:
        for cluster_id in end_map[word_index]:
          coref_list.append("{})".format(cluster_id))
      if word_index in word_map:
        for cluster_id in word_map[word_index]:
          coref_list.append("({})".format(cluster_id))
      if word_index in start_map:
        for cluster_id in start_map[word_index]:
          coref_list.append("({}".format(cluster_id))

      if len(coref_list) == 0:
        row[-1] = "-"
      else:
        row[-1] = "|".join(coref_list)
        #print('start_map', start_map)
        #print('end_map', end_map)
        #print('word_map', word_map)
        #print('coref_list', coref_list)

      #output_file.write("   ".join(row))
      output_file.write("\t".join(row))
      #print('line', line.split('\t'))
      #print('pred', row)
      output_file.write("\n")
      word_index += 1

def official_conll_eval(gold_path, predicted_path, metric, official_stdout=False):
  cmd = ["conll-2012/scorer/v8.01/scorer.pl", metric, gold_path, predicted_path, "none"]
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
  stdout, stderr = process.communicate()
  process.wait()

  stdout = stdout.decode("utf-8")
  if stderr is not None:
    print('stderr:', stderr)

  if official_stdout:
    print("Official result for {}".format(metric))
    print(stdout)

  #print('stdout:', stdout)
  coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
  print('coref_results_match', coref_results_match)
  recall = float(coref_results_match.group(1))
  precision = float(coref_results_match.group(2))
  f1 = float(coref_results_match.group(3))
  return {"r": recall, "p": precision, "f": f1}

def custom_eval(golds, autos, metric):
  p, r, f = metric.evaluate_documents(golds, autos)
  print('%s - %.4f/%.4f/%.4f' % (metric.name, p, r, f))
  return {"r": r, "p": p, "f": f}

def evaluate_conll(gold_path, predictions, log_dir, official_stdout=False):
  test_num = 0
  while os.path.isfile(log_dir + '/out/output_{}.txt'.format(test_num)): test_num += 1
  #with tempfile.NamedTemporaryFile(delete=False, mode="w") as prediction_file:
  with open(log_dir + '/out/output_{}.txt'.format(test_num), mode="w") as prediction_file:
    with open(gold_path, "r") as gold_file:
      output_conll(gold_file, prediction_file, predictions)
    print("Predicted conll file: {}".format(prediction_file.name))
  gold_documents, auto_documents = path2docs(gold_path), path2docs(prediction_file.name)
  p, r, f = MentionEvaluator().evaluate_documents(gold_documents, auto_documents)
  #evalDoc(prediction_file.name, gold_path)
  print('Mention Detection - %.4f/%.4f/%.4f' % (p, r, f))
  return { m.name: custom_eval(gold_documents, auto_documents, m) for m in [BCubeEvaluator(), CeafeEvaluator(), BlancEvaluator()]}
  #return { m: official_conll_eval(gold_file.name, prediction_file.name, m, official_stdout) for m in ("muc", "bcub", "ceafe") }
