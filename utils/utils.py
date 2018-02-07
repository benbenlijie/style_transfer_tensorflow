import scipy.misc as misc
import numpy as np
import os
import tensorflow as tf
slim = tf.contrib.slim


def save_img(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    misc.imsave(path, img)


def scale_img(img, scales):
    ori_shape = img.shape
    new_shape = np.array(ori_shape, dtype=np.int)
    new_shape[:2] = np.multiply(ori_shape[:2], np.array(scales, dtype=np.float))
    return misc.imresize(img, new_shape)


def read_img(path, resize=None):
    img = misc.imread(path, mode="RGB")
    if not (len(img.shape) == 3 and img.shape[-1] == 3):
        img = np.stack([img] * 3)
    if resize:
        img = misc.imresize(img, resize)
    return img


def list_files(path):
    walk = os.walk(path)
    return next(walk)[-1]


def get_network_init_fn(FLAGS):
    tf.logging.info("Use pretrained model {}".format(FLAGS.loss_model_file))
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(",")]
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
        FLAGS.loss_model_file,
        variables_to_restore,
        ignore_missing_vars=True
    )

