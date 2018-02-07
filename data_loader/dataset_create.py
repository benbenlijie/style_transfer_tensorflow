import os
import tensorflow as tf
import sys
import cv2

slim = tf.contrib.slim

dataset_dir = "datasets"

image_folder = "train2014/"

_IMAGE_SIZE = 256

tfrecord_file_name = os.path.join(dataset_dir, "image.tfrecord")

def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
      values: A scalar or list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
      A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
    """Returns a TF-Feature of floats.

    Args:
      values: A scalar of list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width):
    return tf.train.Example(features=tf.train.Features(feature={
        "image/encoded": bytes_feature(image_data),
        "image/format": bytes_feature(image_format),
        "image/height": int64_feature(height),
        "image/width": int64_feature(width),
    }))


def main():
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    filenames = [os.path.join(image_folder, filename)
                 for filename in next(os.walk(image_folder))[-1]]
    with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
        with tf.Graph().as_default():
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_jpeg(image_placeholder)
            with tf.Session() as sess:
                for fname in filenames:
                    sys.stdout.write("\r>> Reading file [%s] image" % fname)
                    sys.stdout.flush()
                    img = cv2.imread(fname, cv2.IMREAD_COLOR)
                    img = img[:, :, ::-1]
                    img = cv2.resize(img, (_IMAGE_SIZE, _IMAGE_SIZE))
                    jpeg_str = sess.run(encoded_image, feed_dict={
                        image_placeholder: img
                    })
                    writer.write(image_to_tfexample(
                        jpeg_str, b'jpeg', _IMAGE_SIZE, _IMAGE_SIZE
                    ))

if __name__ == '__main__':
    main()