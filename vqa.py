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

    self.model_config['frame_num'] = self.clip_num
    with tf.Graph().as_default():
        self.model = GRA(self.model_config, self.clip_num)
        self.model.pretrained_embedding = "VideoQA/" + self.model.pretrained_embedding
        self.model.build_inference()

        sess_config = self.config['session']

        self.sess = tf.Session(config=sess_config)
        save_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        saver = tf.train.Saver()
        if save_path:
            #print('load checkpoint {}.'.format(save_path))
            saver.restore(self.sess, save_path)
        else:
            #print('no checkpoint.')
            return None

  def predict(self, question, cache):
    # If there are no chunks, wait a little
    while cache.newest_id == 0:
        time.sleep(5)

    min_chunk_id = max(cache.newest_id - 4, 0)
    max_chunk_id = cache.newest_id
    for chunk_id in range(max_chunk_id, min_chunk_id, -1):
        self._predict(question, cache.db[chunk_id])

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

    feed_dict = {
        self.model.appear: [chunk.vgg_features],
        self.model.motion: [chunk.c3d_features],
        self.model.question_encode: [question],
    }
    prediction, channel_weight, appear_weight, motion_weight = self.sess.run(
        [self.model.prediction, self.model.channel_weight,
        self.model.appear_weight, self.model.motion_weight],
        feed_dict=feed_dict
    )

    print(chunk.id, prediction, channel_weight, appear_weight, motion_weight)
    prediction = prediction[0]
    channel_weight = channel_weight[0]
    appear_weight = appear_weight[0]
    motion_weight = motion_weight[0]
    answer = self.answerset[prediction]
    print(answer)
    return answer
