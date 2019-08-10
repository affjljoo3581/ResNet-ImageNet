import tensorflow as tf
import dataset
import time


class Logger(object):
    def __init__(self, log_path):
        self.fp = open(log_path, 'a')
        self.last_time_stamp = time.time()
        
        self.batch_metrics = []
    
    def __del__(self):
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
        
        print_msg = '[*] {}/{} \t train ({}) \t validation ({}) \t ETA: {:.1f}'.format(
            current_iters,
            total_iters,
            ', '.join(['{}: {:.4f}'.format(k, v)
                       for k, v in train_metrics.items()]),
            ', '.join(['{}: {:.4f}'.format(k, v)
                       for k, v in val_metrics.items()]),
            eta)
        log_msg = '{}\t{}\t{}\t{}\t{}\n'.format(
            current_iters,
            total_iters,
            '  '.join(['{} {}'.format(k, v)
                       for k, v in train_metrics.items()]),
            '  '.join(['{} {}'.format(k, v)
                       for k, v in val_metrics.items()]),
            eta)
        
        print(print_msg)
        self.fp.write(log_msg)
        self.fp.flush()

def create_train_ops(FLAGS, model_fn):
    iterator = (dataset
                .create_imagenet_train_dataset(FLAGS.imagenet_dataset_dir, FLAGS.train_batch_size)
                .make_one_shot_iterator())
    features, labels = iterator.get_next()
    
    
    


def create_test_ops(FLAGS, model_fn, evaluation=False):
    iterator = (dataset
                .create_imagenet_test_dataset(FLAGS.imagenet_dataset_dir, FLAGS.test_batch_size, evaluation)
                .make_one_shot_iterator())
    
    

def train(FLAGS, model_fn, only_weights=False):
    

def evaluate(FLAGS, model_fn):
    
