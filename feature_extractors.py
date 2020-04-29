import inspect
import numpy as np
import tensorflow as tf
from PIL import Image

from VideoQA.util.c3d import c3d
from VideoQA.util.vgg16 import Vgg16

from pytorch-i3d.extract_features import extract_features

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
        clips = list()

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
                img = frames[clip_start + j]
                img = img.resize((112, 112), Image.BILINEAR)
                frame_data = np.array(img) * 1.0
                frame_data -= self.mean[j]
                new_clip.append(frame_data)
            clips.append(new_clip)

        return clips

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


class ChunkI3DExtractor():
    # TODO: later: might be worth removing i3d extraction and training
    # TMLGA on just c3d features
    def __init__():
        self.max_steps = 64e3
        self.mode = 'rgb'

    def _convert_frames(self, frames):
        # TODO: later: this might not be necessary to do
        # there's probably a way to combine this computation
        # with the frame conversion that happens in the above
        # feature extractors, and avoid doing it twice
        new_frames = []
        for i in range(len(frames)):
            img = frames[i]
            w,h,c = img.shape
            if w < 226 or h < 226:
                d = 226.-min(w,h)
                sc = 1+d/min(w,h)
                img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
            img = (img/255.)*2 - 1
            new_frames.append(img)

        return torch.from_numpy(
            np.asarray(frames, dtype=np.float32).transpose([3, 0, 1, 2])
        )

    def extract(self, frames):
        frames = self._convert_frames(frames)

        i3d = InceptionI3d(400, in_channels=3)
        i3d.replace_logits(157)
        i3d.load_state_dict(torch.load(load_model))

        i3d.train(False)

        # bcthw variable explanation in charades_dataset_full.py > video_to_tensor
        b,c,t,h,w = frames.shape
        if t > 1600:
            features = []
            for start in range(1, t-56, 1600):
                end = min(t-1, start+1600+56)
                start = max(1, start-48)
                ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]), volatile=True)
                features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.numpy())
        else:
            # wrap them in Variable
            inputs = Variable(inputs, volatile=True)
            features = i3d.extract_features(inputs)

        return features