import inspect
import numpy as np
import os
import threading
import tensorflow as tf

from VideoQA.util.c3d import c3d
from VideoQA.util.vgg16 import Vgg16


class ChunkVGGExtractor(object):
    def __init__(self, frame_num, sess, video_size):
        self.frame_num = frame_num
        self.inputs = tf.placeholder(tf.float32, [self.frame_num, 224, 224, 3])
        self.vgg16 = Vgg16()
        self.vgg16.build(self.inputs)
        self.sess = sess
        self.video_size = video_size

    def _select_frames(self, frames):
        """Select representative frames for video.

        Ignore some frames both at begin and end of video.

        Args:
            path: Array of video frames.
        Returns:
            frames: list of frames.
        """
        converted_frames = list()
        # Ignore some frame at begin and end.
        for i in np.linspace(0, self.video_size, self.frame_num + 2)[1:self.frame_num + 1]:
            img = frames[int(i)]
            img = img.resize((224, 224), Image.BILINEAR)
            frame_data = np.array(img)
            converted_frames.append(frame_data)
        return converted_frames

    def extract(self, frames):
        """Get VGG fc7 activations as representation for video.

        Args:
            frames: np array of video frames.
        Returns:
            feature: [batch_size, 4096]
        """
        frames = self._select_frames(frames)
        feature = self.sess.run(
            self.vgg16.relu7, feed_dict={self.inputs: frames})
        return feature


class ChunkC3DExtractor(object):
    def __init__(self, clip_num, sess, frames_per_clip, video_size):
        self.clip_num = clip_num
        self.inputs = tf.placeholder(
            tf.float32, [self.clip_num, frames_per_clip, 112, 112, 3])
        _, self.c3d_features = c3d(self.inputs, 1, clip_num)
        saver = tf.train.Saver()
        path = inspect.getfile(ChunkC3DExtractor)
        path = os.path.abspath(os.path.join(path, os.pardir, "VideoQA/util"))
        saver.restore(sess, os.path.join(
            path, 'sports1m_finetuning_ucf101.model'))
        self.mean = np.load(os.path.join(path, 'crop_mean.npy'))
        self.sess = sess
        self.video_size = video_size
        self.frames_per_clip = frames_per_clip
        
    def _create_clips(self, frames):
        for i in np.linspace(0, self.video_size, self.clip_num + 2)[1:self.clip_num + 1]:
            clip_start = int(i) - int(self.frames_per_clip/2)
            clip_end = int(i) + int(self.frames_per_clip/2) - 1
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > self.video_size:
                clip_start = clip_start - (clip_end - self.video_size)
                clip_end = self.video_size

            new_clip = []
            for j in range(self.frames_per_clip):
                frame_data = frames[clip_start + j]
                img = Image.fromarray(frame_data)
                img = frames[j]
                img = img.resize((112, 112), Image.BILINEAR)
                frame_data = np.array(img) * 1.0
                frame_data -= self.mean[j]
                new_clip.append(frame_data)
            clips.append(new_clip)

    def extract(self, frames):
        """Get 4096-dim activation as feature for video.

        Args:
            path: Video frames.
        Returns:
            feature: [self.batch_size, 4096]
        """
        clips = self._create_clips(frames)
        feature = self.sess.run(
            self.c3d_features, feed_dict={self.inputs: clips})
        return feature


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
