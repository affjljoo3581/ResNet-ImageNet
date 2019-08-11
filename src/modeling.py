import tensorflow as tf
import ops


def bottleneck(x, out_channels, stride=1, is_training=True, name=None):
    with tf.variable_scope(name):
        shortcut = x
        in_channels = x.get_shape().as_list()[1]

        x = ops.relu(ops.bn(x, is_training, name='bn_1'))

        # if the dimension of residual shortcut does not match to the
        # dimension of bottleneck output, do convolution to match its dimension.
        if in_channels != out_channels or stride > 1:
            shortcut = ops.conv(x, out_channels, 1, stride, name='conv_res')

        x = ops.conv(x, out_channels // 4, 1, name='conv_2')
        x = ops.relu(ops.bn(x, is_training, name='bn_3'))
        x = ops.conv(x, out_channels // 4, 3, stride, name='conv_4')
        x = ops.relu(ops.bn(x, is_training, name='bn_5'))
        x = ops.conv(x, out_channels, 1, name='conv_6')

    return x + shortcut

def resblock(x, out_channels, layers=3, stride=1, is_training=True, name=None):
    with tf.variable_scope(name):
        for i in range(1, layers + 1):
            x = bottleneck(x, out_channels, stride if i == 1 else 1, is_training,
                           name='bottleneck_{}'.format(i))

    return x

def ResNet50(x, classes, is_training):
    with tf.variable_scope('ResNet50'):
        with tf.variable_scope('stage_input'):
            x = ops.conv(x, 64, 7, 2, name='conv_1')
            x = ops.pool(x, 2)

        x = resblock(x, 256, 3, 1, is_training, name='stage_1')
        x = resblock(x, 512, 4, 2, is_training, name='stage_2')
        x = resblock(x, 1024, 6, 2, is_training, name='stage_3')
        x = resblock(x, 2048, 3, 2, is_training, name='stage_4')

        with tf.variable_scope('stage_output'):
            x = ops.relu(ops.bn(x, is_training, name='bn_1'))
            x = ops.gap(x)

            x = ops.conv(x, classes, 1, name='conv_2')
            x = ops.squeeze(x)

    return x
