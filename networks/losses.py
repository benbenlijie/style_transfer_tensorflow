# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
from utils.utils import get_network_init_fn
import os

slim = tf.contrib.slim


def gram(layer):
    shape = tf.shape(layer)
    featuremaps = tf.reshape(layer, tf.stack([shape[0], -1, shape[-1]]))
    grams = tf.matmul(featuremaps, featuremaps, transpose_a=True) / tf.to_float(shape[1] * shape[2] * shape[3])
    return grams


def get_style_features(FLAGS):
    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
            FLAGS.loss_model,
            num_classes=1,
            is_training=False
        )
        image_preprocess_fn, _ = preprocessing_factory.get_preprocessing(
            FLAGS.loss_model,
            is_training=False
        )
        size = FLAGS.image_size
        img_bytes = tf.read_file(FLAGS.style_image)
        if FLAGS.style_image.lower().endswith("png"):
            image = tf.image.decode_png(img_bytes)
        else:
            image = tf.image.decode_jpeg(img_bytes)

        image = image_preprocess_fn(image, size, size)
        images = tf.expand_dims(image, 0)

        _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        features = []
        for layer in FLAGS.style_layers:
            feature = endpoints_dict[layer]
            feature = tf.squeeze(gram(feature), [0])
            features.append(feature)

        with tf.Session() as sess:
            init_func = get_network_init_fn(FLAGS)
            init_func(sess)

            if tf.gfile.Exists("generated") is False:
                tf.gfile.MakeDirs("generated")

            return sess.run(features)


def style_loss(style_features, generated_endpoints_dict, style_layers):
    loss = 0
    style_loss_summary = {}
    for style_gram, layer in zip(style_features, style_layers):
        generated_images, _ = tf.split(generated_endpoints_dict[layer], num_or_size_splits=2, axis=0)
        size = tf.size(generated_images)
        layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        style_loss_summary[layer] = layer_style_loss
        loss += layer_style_loss
    return loss, style_loss_summary


def content_loss(generated_endpoints_dict, content_layers):
    loss = 0
    for layer in content_layers:
        generated_image, content_image = tf.split(generated_endpoints_dict[layer],
                                                  num_or_size_splits=2,
                                                  axis=0)
        size = tf.size(generated_image)
        loss += tf.nn.l2_loss(generated_image - content_image) * 2 / tf.to_float(size)
    return loss


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1]))\
        - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1]))\
        - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x))\
        + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss
