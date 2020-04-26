import os
import sys
import tensorflow as tf

from utils import ChunkC3DExtractor, ChunkVGGExtractor


class Chunk(object):

  def __init__(self, cache, id, chunk_size, frames_per_clip):
    # Some constants
    self.frames = []
    self.cache = cache
    self.chunk_size = chunk_size
    self.frames_per_clip = frames_per_clip

    # Chunk number
    self.id = id

    # Computed vgg and c3d features 
    self.vgg_features = list()
    self.c3d_features = list()

    # Image frames of all images in this video chunk
    self.image_frames_for_vgg = list()

  def add_frame(self, frame, frame_count):
    vgg_fram

  def commit(self):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = '0'

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        self.c3d_extractor = ChunkC3DExtractor(self.chunk_size, sess, self.frames_per_clip)
        self.c3d_features = c3d_extractor.extract(video_path)

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        self.vgg_extractor = ChunkVGGExtractor(self.chunk_size, sess, frames_per_clip_c3d)
        self.vgg_features = vgg_extractor.extract(video_path)

    cache.insert(self)
