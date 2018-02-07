import tensorflow as tf
import os

slim = tf.contrib.slim


def get_split(record_file_name, num_sampels, size):
    reader = tf.TFRecordReader

    keys_to_features = {
        "image/encoded": tf.FixedLenFeature((), tf.string, ''),
        "image/format": tf.FixedLenFeature((), tf.string, 'jpeg'),
        "image/height": tf.FixedLenFeature([], tf.int64, tf.zeros([], tf.int64)),
        "image/width": tf.FixedLenFeature([], tf.int64, tf.zeros([], tf.int64)),
    }

    items_to_handlers = {
        "image": slim.tfexample_decoder.Image(shape=[size, size, 3]),
        "height": slim.tfexample_decoder.Tensor("image/height"),
        "width": slim.tfexample_decoder.Tensor("image/width"),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers
    )
    return slim.dataset.Dataset(
        data_sources=record_file_name,
        reader=reader,
        decoder=decoder,
        items_to_descriptions={},
        num_samples=num_sampels
    )


def get_image(num_samples, resize, record_file="image.tfrecord", shuffle=False):
    provider = slim.dataset_data_provider.DatasetDataProvider(
        get_split(record_file, num_samples, resize),
        shuffle=shuffle
    )
    [data_image] = provider.get(["image"])
    return data_image


def batch_images(input_image, batch_size, num_threads=4):
    images = tf.train.batch(
        [input_image],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5 * batch_size
    )
    image_queue = slim.prefetch_queue.prefetch_queue([images], capacity=4)
    return image_queue.dequeue()


if __name__ == '__main__':
    image = get_image(100, 256, "datasets/image.tfrecord")
    import matplotlib.pyplot as plt
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        count = 0
        _, axes = plt.subplots(3, 3)

        try:
            while not coord.should_stop() and count < 9:
                img = sess.run(image)
                axes[count//3, count % 3].imshow(img)

                count += 1

        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)
        plt.show()



