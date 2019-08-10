import tensorflow as tf


def relu(x):
    return tf.nn.relu(x)

def pool(x, size=2):
    return tf.nn.avg_pool(x, size, size, 'SAME', 'NCHW')

def conv(x, out_channels, kernel_size=3, stride=1, name=None)
    in_channels = x.get_shape().as_list()[1]
    
    with tf.variable_scope(name):
        weights = tf.get_variable(
            'weights',
            [kernel_size, kernel_size, in_channels, out_channels],
            tf.float32,
            tf.initializers.he_normal())
    
    return tf.nn.conv2d(x, weights, stride, 'SAME', data_format='NCHW')

def bn(x, is_training=True, name=Nome):
    in_channels = x.get_shape().as_list()[1]
    
    with tf.variable_scope(name):
        scale = tf.get_variable('scale', [in_channels], tf.float32, tf.initializers.ones())
        offset = tf.get_variable('offset', [in_channels], tf.float32, tf.initializers.zeros())
        
        mean = tf.get_variable('mean', [in_channels], tf.float32, tf.initializers.zeros())
        var = tf.get_variable('variance', [in_channels], tf.float32, tf.initializers.zeros())
    
    if is_training:
        x, batch_mean, batch_var = tf.nn.fused_batch_norm(
            x, scale, offset, epsilon=1e-3, data_format='NCHW', is_training=True)
        
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(mean, mean * 0.99 + batch_mean * 0.01))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(var, var * 0.99 + batch_var * 0.01))
    else:
        x, _, _ = tf.nn.fused_batch_norm(
            x, scale, offset, mean, var, epsilon=1e-3, data_format='NCHW', is_training=False)
    
    return x

def gap(x):
    return tf.reduce_mean(x, [2, 3], keepdims=True)

def squeeze(x):
    return tf.reshape(x, [-1, x.get_shape().as_list()[1]])

def softmax_loss(x, y):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=y)
