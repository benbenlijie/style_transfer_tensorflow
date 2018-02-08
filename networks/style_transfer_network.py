import tensorflow as tf
from networks import model, losses
from data_loader.dataset_read import get_image, batch_images
from nets import nets_factory
from preprocessing import preprocessing_factory
import os

slim = tf.contrib.slim


class StyleTransfer:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.g = tf.Graph()
        with self.g.as_default():
            self._init_global_step()
            self._build_model()
            self._init_train_op()
            self._init_saver()
            self._init_summary()

    def _init_saver(self):
        save_variables = []
        for var in tf.global_variables():
            if not var.name.startswith(self.FLAGS.loss_model):
                save_variables.append(var)
        self.model_save_dir = self.FLAGS.model_save_dir or "models"
        self.saver = tf.train.Saver(save_variables)

    def _init_summary(self):
        summary_dir = self.FLAGS.log_dir or "logs/"
        if not tf.gfile.Exists(summary_dir):
            tf.gfile.MakeDirs(summary_dir)
        self.summary = slim.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(logdir=summary_dir, graph=self.g)

    def _init_global_step(self):
        self.global_step = tf.train.get_or_create_global_step()

    def _build_model(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        style_features = losses.get_style_features(self.FLAGS)

        network_fn = nets_factory.get_network_fn(
            self.FLAGS.loss_model,
            num_classes=1,
            is_training=False
        )
        preprocessing_fn, unprocessing_fn = preprocessing_factory.get_preprocessing(
            self.FLAGS.loss_model,
            is_training=False
        )
        image = get_image(self.FLAGS.num_samples, self.FLAGS.image_size, self.FLAGS.tfrecord_file)
        processed_image = preprocessing_fn(image, self.FLAGS.image_size, self.FLAGS.image_size)
        images = batch_images(processed_image, batch_size=self.FLAGS.batch_size)
        generated = model.net(images)
        self.generated = generated
        """prepare for evaluate the loss"""
        processed_generated = [
            preprocessing_fn(image, self.FLAGS.image_size, self.FLAGS.image_size)
            for image in tf.unstack(generated, axis=0, num=self.FLAGS.batch_size)
        ]
        processed_generated = tf.stack(processed_generated)
        _, endpoints_dict = network_fn(tf.concat([processed_generated, images], 0),
                                       spatial_squeeze=False)
        """losses"""
        style_loss, style_loss_summary = losses.style_loss(style_features, endpoints_dict, self.FLAGS.style_layers)
        content_loss = losses.content_loss(endpoints_dict, self.FLAGS.content_layers)
        variation_loss = losses.total_variation_loss(generated)

        total_loss = self.FLAGS.style_weight * style_loss + \
                     self.FLAGS.content_weight * content_loss + \
                     self.FLAGS.variation_weight * variation_loss
        self.style_loss = tf.identity(style_loss, "style_loss")
        self.content_loss = tf.identity(content_loss, "content_loss")
        self.variation_loss = tf.identity(variation_loss, "variation_loss")
        self.total_loss = tf.identity(total_loss, "total_loss")
        tf.losses.add_loss(self.style_loss)
        tf.losses.add_loss(self.content_loss)
        tf.losses.add_loss(self.variation_loss)
        tf.losses.add_loss(self.total_loss)

        slim.summarize_collection(tf.GraphKeys.LOSSES)
        # slim.summarize_variables("style_transfer")
        slim.summary.image("generated", generated)
        slim.summary.image("origin", tf.stack([
            unprocessing_fn(image) for image in tf.unstack(images, axis=0, num=self.FLAGS.batch_size)
        ]))

    def _init_train_op(self):
        train_variables = []
        for var in tf.trainable_variables():
            if not var.name.startswith(self.FLAGS.loss_model):
                train_variables.append(var)
        self.learning_rate = tf.train.exponential_decay(
            self.FLAGS.learning_rate, self.global_step, 1000, 0.66, name="learning_rate")
        slim.summarize_tensor(self.learning_rate)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss,
                                                                            global_step=self.global_step,
                                                                            var_list=train_variables)

    def restore_network(self, sess):
        init_fn = self.get_network_init_fn()
        init_fn(sess)
        last_file = tf.train.latest_checkpoint(self.model_save_dir)
        if last_file:
            tf.logging.info("restore model from {}".format(last_file))
            self.saver.restore(sess, last_file)

    def save_network(self, sess, global_step=None):
        self.saver.save(sess, os.path.join(self.model_save_dir, "style_transfer.ckpt"), global_step=global_step)

    def get_network_init_fn(self):
        tf.logging.info("Use pretrained model {}".format(self.FLAGS.loss_model_file))
        exclusions = []
        if self.FLAGS.checkpoint_exclude_scopes:
            exclusions = [scope.strip()
                          for scope in self.FLAGS.checkpoint_exclude_scopes.split(",")]
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        return slim.assign_from_checkpoint_fn(
            self.FLAGS.loss_model_file,
            variables_to_restore,
            ignore_missing_vars=True
        )

    def evaluate_network(self, image_path, image_size=None):
        with tf.Graph().as_default() as g:
            filename_queue = tf.train.string_input_producer([image_path])
            reader = tf.WholeFileReader()
            _, value = reader.read(filename_queue)
            if image_path.endswith("png"):
                test_image = tf.image.decode_png(value)
            elif image_path.endswith("jpeg"):
                test_image = tf.image.decode_jpeg(value)
            else:
                test_image = tf.image.decode_image(value)

            preprocessing_fn, unprocessing_fn = preprocessing_factory.get_preprocessing(
                self.FLAGS.loss_model,
                is_training=False
            )
            image_size = image_size or [self.FLAGS.image_size, self.FLAGS.image_size]
            processed_image = preprocessing_fn(test_image, image_size[0], image_size[1])
            images = tf.expand_dims(processed_image, axis=0)
            generated = model.net(images)
            generated = tf.squeeze(generated, axis=0)
            with tf.Session(graph=g) as sess:
                last_file = tf.train.latest_checkpoint(self.model_save_dir)
                if last_file:
                    tf.logging.info("restore model from {}".format(last_file))
                    saver = tf.train.Saver()
                    saver.restore(sess, last_file)
                tf.logging.info("start to transfer")
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                styled_image = sess.run(generated)
                coord.request_stop()
                coord.join(threads)
            tf.logging.info("finish transfer")
            return styled_image

