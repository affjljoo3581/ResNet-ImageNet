import tensorflow as tf
import os


def decode_and_crop_image(image_buffer, bbox):
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)

    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    cropped = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=3)
    cropped = tf.image.random_flip_left_right(cropped)

    return cropped

def aspect_preserving_resize_and_crop_image(image, resize_min, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)
    resize_min = tf.cast(resize_min, tf.float32)

    scale_ratio = resize_min / tf.minimum(height, width)
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    crop_top = (new_height - crop_height) // 2
    crop_left = (new_width - crop_width) // 2

    image = tf.image.resize(image, [new_height, new_width])
    image = tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

    return image

def parse_dataset_example(serialized):
    feature_map = {
        'image/encoded':          tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label':      tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':       tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32)
    }
    features = tf.io.parse_single_example(serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32) - 1

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox

def process_dataset_image(serialized, is_training=True):
    image_buffer, label, bbox = parse_dataset_example(serialized)

    if is_training:
        image = decode_and_crop_image(image_buffer, bbox)
        image = tf.image.resize(image, [224, 224])
    else:
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = aspect_preserving_resize_and_crop_image(image, 256, 224, 224)
    image = tf.reshape(tf.cast(image, tf.float32), [224, 224, 3])
    image = 2 * tf.transpose(image, [2, 0, 1]) - 1

    return image, label

def create_imagenet_train_dataset(dataset_dir, batch_size):
    filenames = [os.path.join(dataset_dir, 'train-{:05d}-of-01024'.format(i))
                 for i in range(1024)]
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.shuffle(1024)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(batch_size)
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat()
    dataset = dataset.map(
        lambda serialized: process_dataset_image(serialized, is_training=True),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def create_imagenet_test_dataset(dataset_dir, batch_size, evaluation=False):
    filenames = [os.path.join(dataset_dir, 'validation-{:05d}-of-00128'.format(i))
                 for i in range(128)]
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(1 if evaluation else -1)
    dataset = dataset.map(
        lambda serialized: process_dataset_image(serialized, is_training=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
