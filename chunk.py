import os
import sys

import tensorflow as tf

class Chunk(object):

  def __init__(self, cache, id, c3d_extractor, vgg_extractor):
    self.frames = []
    self.cache = cache

    self.id = id

    self.vgg_features = list()
    self.c3d_features = list()
    self.c3d_extractor = c3d_extractor
    self.vgg_extractor = vgg_extractor

  def add_frame(self, frame, frame_count):
    pass

  def commit(self):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = '0'

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        c3d_extractor.begin_session()
        self.c3d_features = c3d_extractor.extract(video_path)

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        vgg_extractor.begin_session()
        self.vgg_features = vgg_extractor.extract(video_path)

    cache.insert(self)
