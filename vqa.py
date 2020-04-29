from VideoQA.model.gra import GRA
import VideoQA.config as cfg

class VQA:
  def __init__(self, config, checkpoint_dir):
    self.config = cfg.get('gra', 'data/msvd_qa', config, None)
    self.model_config = self.config['model']
    self.sess_config = self.config['session']
    self.checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoint')
    self.answerset = pd.read_csv(os.path.join(
      self.config['preprocess_dir'], 'answer_set.txt'
    ), header=None)[0]

  def predict(self, question, chunk):
    with tf.Graph().as_default():
      model = GRA(model_config)
      model.build_inference()

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
        prediction = prediction[0]
        channel_weight = channel_weight[0]
        appear_weight = appear_weight[0]
        motion_weight = motion_weight[0]
        answer = answerset[prediction]

    return answer