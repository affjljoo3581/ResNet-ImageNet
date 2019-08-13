from absl import flags, app

import tensorflow as tf
import models
import utils
import os


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', 'gs://my-dl-dataset/imagenet', 'ImageNet dataset path')
flags.DEFINE_integer('classes', 1000, 'number of ImageNet dataset classes')

flags.DEFINE_string('model_path', '../result/base', 'path of model weights to save or restore')
flags.DEFINE_string('log_path', '../train_base_model.log', 'ImageNet dataset path')

flags.DEFINE_integer('train_batch_size', 64, 'train batch size')
flags.DEFINE_integer('test_batch_size', 256, 'test and evaluation batch size')

flags.DEFINE_float('momentum', 0.9, 'momentum factor of optimizer')
flags.DEFINE_float('max_lr', 1e-1, 'maximum learning rate')
flags.DEFINE_float('min_lr', 1e-4, 'minimum learning rate')

flags.DEFINE_bool('use_fp16', True, 'use half-precision to decrease training time and memory usage')
flags.DEFINE_float('loss_scale', 128, 'loss scaling factor to preserve small gradient magnitudes')

flags.DEFINE_integer('total_iters', 20 * 1281167 // 64, 'number of total iterations for training')
flags.DEFINE_integer('log_iters', 500, 'number of iterations to test model and log results')
flags.DEFINE_integer('save_iters', 5000, 'number of iterations to save model weights')


def main(argv):
    model_fn = lambda x, is_training: models.ResNet152(x, FLAGS.classes, is_training)

    utils.train(FLAGS, model_fn)
    tf.reset_default_graph()
    metrics = utils.evaluate(FLAGS, model_fn)

    with open(FLAGS.log_path, 'a') as fp:
        fp.write(str(metrics))

    os.system('sudo poweroff')

if __name__ == '__main__':
    app.run(main)
