import tensorflow as tf
import dataset, ops
import shutil
import time
import os


class Logger(object):
    def __init__(self, log_path):
        self.fp = open(log_path, 'a')
        self.last_time_stamp = time.time()

        self.batch_metrics = []

    def __del__(self):
        self.fp.close()

    def close(self):
        self.fp.close()

    def timer(self):
        eta = time.time() - self.last_time_stamp
        self.last_time_stamp = time.time()

        return eta

    def add_batch(self, metrics):
        self.batch_metrics.append(metrics)

    def zip_batch(self):
        metrics = {}
        for batch in self.batch_metrics:
            for k, v in batch.items():
                if k not in metrics: metrics[k] = []
                metrics[k].append(v)

        self.batch_metrics = []
        return {k: sum(v) / len(v) for k, v in metrics.items()}

    def __call__(self, current_iters, total_iters, val_metrics):
        eta = self.timer()
        train_metrics = self.zip_batch()

        print_msg = '[*] {0}/{1} \t train ({3}) \t validation ({4}) \t ETA: {2:.1f}'.format(
            current_iters, total_iters, eta,
            ', '.join(['{}: {:.4f}'.format(k, v) for k, v in train_metrics.items()]),
            ', '.join(['{}: {:.4f}'.format(k, v) for k, v in val_metrics.items()]))
        log_msg = '{0}\t{1}\t{3}\t{4}\t{2}\n'.format(
            current_iters, total_iters, eta,
            '  '.join(['{} {}'.format(k, v) for k, v in train_metrics.items()]),
            '  '.join(['{} {}'.format(k, v) for k, v in val_metrics.items()]))

        print(print_msg)
        self.fp.write(log_msg)
        self.fp.flush()

def create_train_ops(FLAGS, model_fn):
    iterator = (dataset
                .create_imagenet_train_dataset(FLAGS.dataset_dir, FLAGS.train_batch_size)
                .make_one_shot_iterator())
    features, labels = iterator.get_next()

    if FLAGS.use_fp16:
        # cast the input to float16 and the output to float32
        features = tf.cast(features, tf.float16)

        logits = model_fn(features, is_training=True)
        logits = tf.cast(logits, tf.float32)
    else:
        logits = model_fn(features, is_training=True)

    # calculate the loss and accuracy metrics
    loss_op = ops.softmax_loss(logits, labels)
    accuracy_1_op = ops.accuracy(logits, labels, top_k=1)
    accuracy_5_op = ops.accuracy(logits, labels, top_k=5)

    with tf.variable_scope('optimization'):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.cosine_decay(FLAGS.max_lr, global_step, FLAGS.total_iters, FLAGS.min_lr / FLAGS.max_lr)

        optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            if FLAGS.use_fp16:
                # use loss scaling to preserve small gradient magnitudes
                grad_and_vars = optimizer.compute_gradients(loss_op * FLAGS.loss_scale)
                grad_and_vars = [(grad / FLAGS.loss_scale, var) for grad, var in grad_and_vars if grad is not None]
            else:
                grad_and_vars = optimizer.compute_gradients(loss_op)

            # apply gradients to each variables
            train_op = optimizer.apply_gradients(grad_and_vars, global_step)

    return train_op, {'loss': loss_op, 'top-1 accuracy': accuracy_1_op, 'top-5 accuracy': accuracy_5_op}

def create_test_ops(FLAGS, model_fn, evaluation=False):
    iterator = (dataset
                .create_imagenet_test_dataset(FLAGS.dataset_dir, FLAGS.test_batch_size, evaluation)
                .make_one_shot_iterator())
    features, labels = iterator.get_next()

    # do not use the half-precision for precise evaluation
    logits = model_fn(features, is_training=True)

    # calculate the loss and accuracy metrics
    loss_op = ops.softmax_loss(logits, labels)
    accuracy_1_op = ops.accuracy(logits, labels, top_k=1)
    accuracy_5_op = ops.accuracy(logits, labels, top_k=5)

    return loss_op, {'loss': loss_op, 'top-1 accuracy': accuracy_1_op, 'top-5 accuracy': accuracy_5_op}

def train(FLAGS, model_fn, only_weights=False):
    train_ops = create_train_ops(FLAGS, model_fn)

    tf.get_variable_scope().reuse_variables()
    test_ops = create_test_ops(FLAGS, model_fn, evaluation=False)

    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())

        # restore weights if there is a checkpoint file
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.model_path)
        if latest_ckpt:
            var_list = [v for v in tf.trainable_variables() if not v.name.startswith('optimization')]
            tf.train.Saver(var_list if only_weights else None).restore(sess, latest_ckpt)

        logger = Logger(FLAGS.log_path)
        latest_step = sess.run(tf.train.get_global_step())
        for step in range(latest_step + 1, FLAGS.total_iters + 1):
            _, metrics = sess.run(train_ops)
            logger.add_batch(metrics)

            # get metrics for test dataset and log the training results
            if step % FLAGS.log_iters == 0:
                logger(step, FLAGS.total_iters, sess.run(test_ops))

            # save the model to resume the training after stopped
            if step % FLAGS.save_iters == 0:
                if os.path.exists(FLAGS.model_path): shutil.rmtree(FLAGS.model_path)
                tf.train.Saver().save(sess, os.path.join(FLAGS.model_path, 'model.ckpt'), step)

        logger.close()
        if os.path.exists(FLAGS.model_path): shutil.rmtree(FLAGS.model_path)

        var_list = [v for v in tf.trainable_variables() if not v.name.startswith('optimization')]
        tf.train.Saver(var_list).save(sess, os.path.join(FLAGS.model_path, 'model.ckpt'))

def evaluate(FLAGS, model_fn):
    eval_ops = create_test_ops(FLAGS, model_fn, evaluation=True)

    total_metrics = {}
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, os.path.join(FLAGS.model_path, 'model.ckpt'))

        try:
            while True:
                for k, v in sess.run(eval_ops).items():
                    if k not in total_metrics: total_metrics[k] = []
                    total_metrics[k].append(v)
        except tf.errors.OutOfRangeError: pass

    return {k: sum(v) / len(v) for k, v in total_metrics.items()}
