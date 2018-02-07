import tensorflow as tf
slim = tf.contrib.slim


def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def deconv2d(x, num_outputs, kernel_size, stride, activation_fn):
    # 1. resize image
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    new_height = height * stride * 2
    new_width = width * stride * 2
    x_resized = tf.image.resize_images(x, [new_height, new_width],
                                       tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = slim.conv2d(x_resized, num_outputs=num_outputs,
                    kernel_size=kernel_size, stride=stride,
                    activation_fn=None)
    x = instance_norm(x)
    x = activation_fn(x)
    return x


def net(inputs, training):
    with tf.variable_scope("style_transfer"):
        x = tf.pad(inputs, [[0, 0], [10, 10], [10, 10], [0, 0]], mode="REFLECT")

        with tf.variable_scope("conv1"):
            x = slim.conv2d(x, num_outputs=32, kernel_size=9, stride=1, activation_fn=None)
            x = instance_norm(x)
            x = tf.nn.relu(x)
        with tf.variable_scope("conv2"):
            x = slim.conv2d(x, num_outputs=64, kernel_size=3, stride=2, activation_fn=None)
            x = instance_norm(x)
            x = tf.nn.relu(x)
        with tf.variable_scope("conv3"):
            x = slim.conv2d(x, num_outputs=128, kernel_size=3, stride=2, activation_fn=None)
            x = instance_norm(x)
            x = tf.nn.relu(x)
        with tf.variable_scope("residual1"):
            res = slim.repeat(x, 2, slim.conv2d, num_outputs=128, kernel_size=3, stride=1)
            x = x + res
        with tf.variable_scope("residual2"):
            res = slim.repeat(x, 2, slim.conv2d, num_outputs=128, kernel_size=3, stride=1)
            x = x + res
        with tf.variable_scope("residual3"):
            res = slim.repeat(x, 2, slim.conv2d, num_outputs=128, kernel_size=3, stride=1)
            x = x + res
        with tf.variable_scope("residual4"):
            res = slim.repeat(x, 2, slim.conv2d, num_outputs=128, kernel_size=3, stride=1)
            x = x + res
        with tf.variable_scope("residual5"):
            res = slim.repeat(x, 2, slim.conv2d, num_outputs=128, kernel_size=3, stride=1)
            x = x + res
        with tf.variable_scope("deconv1"):
            x = deconv2d(x, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        with tf.variable_scope("deconv2"):
            x = deconv2d(x, num_outputs=32, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        with tf.variable_scope("deconv3"):
            x = deconv2d(x, num_outputs=3, kernel_size=9, stride=1, activation_fn=tf.nn.tanh)

        x = (x + 1) * (255.0 / 2)

        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        return tf.slice(x, [0, 10, 10, 0], tf.stack([-1, height-20, width-20, -1]))
