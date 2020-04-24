import inspect
import numpy as np
import os
import threading
import tensorflow as tf

from VideoQA.util.c3d import c3d
from VideoQA.util.vgg16 import Vgg16


class ChunkVGGExtractor(object):
    def __init__(self, frame_num, sess):
        self.frame_num = frame_num
        self.inputs = tf.placeholder(tf.float32, [self.frame_num, 224, 224, 3])
        self.vgg16 = Vgg16()
        self.vgg16.build(self.inputs)
        self.sess = sess

    def _select_frames(self, path):
        """Select representative frames for video.

        Ignore some frames both at begin and end of video.

        Args:
            path: Path of video.
        Returns:
            frames: list of frames.
        """
        frames = list()
        # video_info = skvideo.io.ffprobe(path)
        video_data = skvideo.io.vread(path)
        total_frames = video_data.shape[0]
        # Ignore some frame at begin and end.
        for i in np.linspace(0, total_frames, self.frame_num + 2)[1:self.frame_num + 1]:
            frame_data = video_data[int(i)]
            img = Image.fromarray(frame_data)
            img = img.resize((224, 224), Image.BILINEAR)
            frame_data = np.array(img)
            frames.append(frame_data)
        return frames

    def extract(self, path):
        """Get VGG fc7 activations as representation for video.

        Args:
            path: Path of video.
        Returns:
            feature: [batch_size, 4096]
        """
        frames = self._select_frames(path)
        # We usually take features after the non-linearity, by convention.
        feature = self.sess.run(
            self.vgg16.relu7, feed_dict={self.inputs: frames})
        return feature


class ChunkC3DExtractor(object):
    def __init__(self, clip_num, sess):
        self.clip_num = clip_num
        self.inputs = tf.placeholder(
            tf.float32, [self.clip_num, 16, 112, 112, 3])
        _, self.c3d_features = c3d(self.inputs, 1, clip_num)
        saver = tf.train.Saver()
        path = inspect.getfile(ChunkC3DExtractor)
        path = os.path.abspath(os.path.join(path, os.pardir, "VideoQA/util"))
        saver.restore(sess, os.path.join(
            v, 'sports1m_finetuning_ucf101.model'))
        self.mean = np.load(os.path.join(path, 'crop_mean.npy'))
        self.sess = sess
        
    def _select_clips(self, path):
        """Select self.batch_size clips for video. Each clip has 16 frames.

        Args:
            path: Path of video.
        Returns:
            clips: list of clips.
        """
        clips = list()
        # video_info = skvideo.io.ffprobe(path)
        video_data = skvideo.io.vread(path)
        total_frames = video_data.shape[0]
        height = video_data[1]
        width = video_data.shape[2]
        for i in np.linspace(0, total_frames, self.clip_num + 2)[1:self.clip_num + 1]:
            # Select center frame first, then include surrounding frames
            clip_start = int(i) - 8
            clip_end = int(i) + 8
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > total_frames:
                clip_start = clip_start - (clip_end - total_frames)
                clip_end = total_frames
            clip = video_data[clip_start:clip_end]
            new_clip = []
            for j in range(16):
                frame_data = clip[j]
                img = Image.fromarray(frame_data)
                img = img.resize((112, 112), Image.BILINEAR)
                frame_data = np.array(img) * 1.0
                frame_data -= self.mean[j]
                new_clip.append(frame_data)
            clips.append(new_clip)
        return clips

    def extract(self, path):
        """Get 4096-dim activation as feature for video.

        Args:
            path: Path of video.
        Returns:
            feature: [self.batch_size, 4096]
        """
        clips = self._select_clips(path)
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
