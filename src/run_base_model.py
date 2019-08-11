from absl import flags, app

import tensorflow as tf
import optimization
import modeling
import os


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', 'gs://my-dl-dataset/imagenet', 'ImageNet dataset path')
flags.DEFINE_integer('classes', 1000, 'number of ImageNet dataset classes')

flags.DEFINE_string('model_path', '../result/base', 'path of model weights to save or restore')
flags.DEFINE_string('log_path', 'gs://my-dl-dataset/imagenet', 'ImageNet dataset path')

flags.DEFINE_integer('train_batch_size', 32, 'train batch size')
flags.DEFINE_integer('test_batch_size', 256, 'test and evaluation batch size')

flags.DEFINE_float('momentum', 0.9, 'momentum factor of optimizer')
flags.DEFINE_float('max_lr', 1e-2, 'maximum learning rate')
flags.DEFINE_float('min_lr', 1e-5, 'minimum learning rate')

flags.DEFINE_bool('use_fp16', True, 'use half-precision to decrease training time and memory usage')
flags.DEFINE_float('loss_scale', 128, 'loss scaling factor to preserve small gradient magnitudes')

flags.DEFINE_integer('total_iters', 20 * 1281167 // 64, 'number of total iterations for training')
flags.DEFINE_integer('log_iters', 100, 'number of iterations to test model and log results')
flags.DEFINE_integer('save_iters', 5000, 'number of iterations to save model weights')


def main(argv):
    model_fn = lambda x, is_training: modeling.ResNet50(x, FLAGS.classes, is_training)

    optimization.train(FLAGS, model_fn)
    tf.reset_default_graph()
    metrics = optimization.evaluate(FLAGS, model_fn)

    with open(FLAGS.log_path, 'a') as fp:
        fp.write(str(metrics))

    os.system('sudo poweroff')

if __name__ == '__main__':
    app.run(main)
