from VideoQA.model.gra import GRA
import VideoQA.config as cfg
import os
import pandas as pd
import tensorflow as tf
import time

class VQA:
  def __init__(self, config, checkpoint_dir, vocab_path, clip_num):
    self.config = cfg.get('gra', 'msvd_qa', config, None)
    self.model_config = self.config['model']
    self.sess_config = self.config['session']
    self.checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoint')
    self.answerset = pd.read_csv(os.path.join(
      "VideoQA/", self.config['preprocess_dir'], 'answer_set.txt'
    ), header=None)[0]
    self.vocab = pd.read_csv(vocab_path, header=None)[0]
    self.clip_num = clip_num

  def predict(self, question, cache):
    # If there are no chunks, wait a little
    while cache.newest_id == 0:
        time.sleep(5)

    return self._predict(question, cache.db[1])
    #for chunk_id in range(cache.newest_id, cache.oldest_id, -1):
    #    self._predict(question, cache.db[chunk_id])

  def _encode_question(self, question):
    """Map question to sequence of vocab id. 3999 for word not in vocab."""
    question_id = ''
    words = question.rstrip('?').split()
    for word in words:
        if word in self.vocab.values:
            question_id = question_id + \
                str(self.vocab[self.vocab == word].index[0]) + ','
        else:
            question_id = question_id + '3999' + ','

    question_id = question_id.rstrip(',')
    question = [int(x) for x in question_id.split(',')]
    return question

  def _predict(self, question, chunk):
    question = self._encode_question(question)

    self.model_config['frame_num'] = chunk.clip_num
    with tf.Graph().as_default():
      model = GRA(self.model_config, chunk.clip_num)
      model.pretrained_embedding = "VideoQA/" + model.pretrained_embedding
      model.build_inference()

      sess_config = self.config['session']

      with tf.Session(config=sess_config) as sess:
        save_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        saver = tf.train.Saver()
        if save_path:
            print('load checkpoint {}.'.format(save_path))
            saver.restore(sess, save_path)
        else:
            print('no checkpoint.')
            return None

        feed_dict = {
            model.appear: [chunk.vgg_features],
            model.motion: [chunk.c3d_features],
            model.question_encode: [question],
        }
        prediction, channel_weight, appear_weight, motion_weight = sess.run(
            [model.prediction, model.channel_weight,
            model.appear_weight, model.motion_weight],
            feed_dict=feed_dict
        )

        print(chunk.id, prediction, channel_weight, appear_weight, motion_weight)
        prediction = prediction[0]
        channel_weight = channel_weight[0]
        appear_weight = appear_weight[0]
        motion_weight = motion_weight[0]
        answer = self.answerset[prediction]
        return answer
