import tensorflow as tf
from networks.style_transfer_network import StyleTransfer
import time


class StyleTransferTrainer:
    def __init__(self, sess: tf.Session, model: StyleTransfer):
        self.sess = sess
        self.model = model
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        model = self.model
        model.restore_network(self.sess)
        tf.logging.info("start to train")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                start_time = time.time()
                _, loss, step = self.sess.run([model.train_op, model.total_loss, model.global_step])
                elapsed_time = time.time() - start_time
                if step % 10 == 0:
                    tf.logging.info("step {}, total loss {}, secs/step {}".format(step,
                                                                                  loss,
                                                                                  elapsed_time))
                if step % 25 == 0:
                    summary_str = self.sess.run(model.summary)
                    model.summary_writer.add_summary(summary_str, step)
                    model.summary_writer.flush()
                if step % 500 == 0:
                    model.save_network(self.sess, global_step=step)
                pass
        except tf.errors.OutOfRangeError as e:
            model.save_network(self.sess)
            tf.logging.info("finish training")
        finally:
            coord.request_stop()
        coord.join(threads)
