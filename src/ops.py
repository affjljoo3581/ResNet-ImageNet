import tensorflow as tf


def relu(x):
    return tf.nn.relu(x)

def pool(x, size=2):
    return tf.nn.avg_pool(x, size, size, 'SAME', 'NCHW')

def conv(x, out_channels, kernel_size=3, stride=1, name=None):
    in_channels = x.get_shape().as_list()[1]

    # define the weight variable for convolution operation.
    with tf.variable_scope(name):
        weights = tf.get_variable(
            'weights',
            [kernel_size, kernel_size, in_channels, out_channels],
            tf.float32,
            tf.initializers.he_normal())

    # cast to the same dtype for convolution operation.
    if x.dtype != weights.dtype: weights = tf.cast(weights, x.dtype)

    return tf.nn.conv2d(x, weights, stride, 'SAME', data_format='NCHW')

def bn(x, is_training=True, name=None):
    original_dtype = x.dtype
    in_channels = x.get_shape().as_list()[1]

    # cast to float32 dtype because tf.nn.fused_batch_norm only supports float32.
    if x.dtype != tf.float32: x = tf.cast(x, tf.float32)

    # define variables for batch-normalization.
    with tf.variable_scope(name):
        scale = tf.get_variable('scale', [in_channels], tf.float32, tf.initializers.ones())
        offset = tf.get_variable('offset', [in_channels], tf.float32, tf.initializers.zeros())

        mean = tf.get_variable('mean', [in_channels], tf.float32, tf.initializers.zeros())
        var = tf.get_variable('variance', [in_channels], tf.float32, tf.initializers.zeros())

    if is_training:
        # do batch-normalization and update total mean and variance.
        x, batch_mean, batch_var = tf.nn.fused_batch_norm(
            x, scale, offset, epsilon=1e-3, data_format='NCHW', is_training=True)

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(mean, mean * 0.99 + batch_mean * 0.01))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(var, var * 0.99 + batch_var * 0.01))
    else:
        # do batch-normalization with calculated total mean and variance.
        x, _, _ = tf.nn.fused_batch_norm(
            x, scale, offset, mean, var, epsilon=1e-3, data_format='NCHW', is_training=False)

    # cast x to the original dtype.
    if x.dtype != original_dtype: x = tf.cast(x, original_dtype)

    return x

def gap(x):
    return tf.reduce_mean(x, [2, 3], keepdims=True)

def squeeze(x):
    return tf.reshape(x, [-1, x.get_shape().as_list()[1]])

def softmax_loss(x, y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=y))

def accuracy(x, y, top_k=1):
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(x, y, top_k), tf.float32))