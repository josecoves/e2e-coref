#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pathlib
import tensorflow as tf
import coref_model as cm
import util
import numpy
numpy.set_printoptions(threshold=2000)

if __name__ == "__main__":
  print('tf.__version__', tf.__version__)
  config = util.initialize_from_env()

  Debug = config["debug"]
  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]
  if Debug:
    eval_frequency = 1
    report_frequency = 1
  print(eval_frequency)

  #model = cm.CorefModel(config)
  #saver = tf.train.Saver()

  log_dir = config["log_dir"]
  pathlib.Path(log_dir + "/out").mkdir(parents=True, exist_ok=True)
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_f1 = 0



  with tf.Session() as session:
    model = cm.CorefModel(config, session)
    saver = tf.train.Saver()

    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0

    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)

    initial_time = time.time()
    while True:
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
      accumulated_loss += tf_loss

      """
      if False and Debug:
        print('tf_loss',tf_loss)
        print('tf_global_step',tf_global_step)
        print('predictions')
        for t in session.run(model.predictions):
          print(t.shape)
          print(t)
        for k,v in model.debug.items():
          if k != 'k,c':
            print(k)
            #print(session.run(tf.shape(v)), v.get_shape().as_list())
            #print('first:', session.run(v[0]))
            t = session.run(v)
            print(t.shape, v)
            print(t)
            #print(session.run(v))
        #exit(3)"""
      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0

      if tf_global_step % eval_frequency == 0:
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        eval_summary, eval_f1 = model.evaluate(session, log_dir, official_stdout=True, test=False)

        if eval_f1 > max_f1:
          max_f1 = eval_f1
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

        writer.add_summary(eval_summary, tf_global_step)
        writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

        print("[{}] f1_eval={:.3f}, f1_max={:.3f}".format(tf_global_step, eval_f1, max_f1))

        if eval_frequency == 1: exit()
