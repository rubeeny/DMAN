#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import logging
import tensorflow as tf

sys.path.append("../../")
from model import Model
from rank_util import load_raw_data, prepare_dataset, BatchManager


flags = tf.flags

# Data Configuration
flags.DEFINE_string('log_path', './run_eta_tf.log', 'Path of log.')
flags.DEFINE_string('raw_train_dataset', './data/sample.csv', 'Path to train set .csv file.')
flags.DEFINE_string('raw_dev_dataset', './data/sample.csv', 'Path to valid set .csv file.')
flags.DEFINE_string('save_parameters',"./data/model","model save path")
flags.DEFINE_string('tensorboard_dir',"./data/model/tensorboard","model save path")

flags.DEFINE_integer('isrestore',0,'restore from trained')
flags.DEFINE_integer('filter_num',0,'restore from trained')
flags.DEFINE_integer('use_multi_task',1,'use multi-task framework or not')
flags.DEFINE_float('init_best',sys.maxint*1.0,'init best')
flags.DEFINE_integer('is_filter_train',1,'filter data in train set or not')
flags.DEFINE_integer('is_filter_test',1,'filter data in test set or not')
flags.DEFINE_integer('select_more',1,'select the sample which order number >= filter_order_num')
flags.DEFINE_integer('filter_order_num',2,'select the sample which order number equal to filter_order_num')
flags.DEFINE_integer('random_shuffle_train',0,'shuffle orders or not in train set')
flags.DEFINE_integer('random_shuffle_test',0,'shuffle orders or not in test set')

# Network Configuration
flags.DEFINE_integer('d_model',128,'number of hidden_size')
flags.DEFINE_integer('d_ff',128,'ff_hidden_size')
flags.DEFINE_integer('max_length',10,'max_length')
flags.DEFINE_integer('num_blocks',2,'num_blocks')
flags.DEFINE_integer('num_heads',4,'num_blocks')
flags.DEFINE_float('dropout_rate',0.5,'num_blocks')
flags.DEFINE_float('l2_scale',100,'l2_scale')
flags.DEFINE_float('alpha', 0.5, 'auxiliary task weight in multi-task learning framework')
flags.DEFINE_float('threshold_1', -10, 'used for customizing eta loss, see function customized_loss in layers.py')
flags.DEFINE_float('threshold_2', 10, 'used for customizing eta loss, see function customized_loss in layers.py')
flags.DEFINE_float('penalty_1', 1.2, 'used for customizing eta loss, see function customized_loss in layers.py')
flags.DEFINE_float('penalty_2', 1.8, 'used for customizing eta loss, see function customized_loss in layers.py')
flags.DEFINE_integer('is_use_weight', 0, 'is_use_weigth to log loss')
flags.DEFINE_integer('is_use_log_loss', 0, 'is_use_log_loss')
flags.DEFINE_integer('is_use_evaluation_layer', 1, 'is_use_log_loss')
flags.DEFINE_integer('is_use_eta_attention_layer', 1, 'is_use_log_loss')
flags.DEFINE_float('main_task_weight', 1, 'main_task_weight')
flags.DEFINE_float('rank_task_weight', 5, 'rank_task_weight')
flags.DEFINE_float('consistent_weight', 20, 'consistent_weight')
flags.DEFINE_float('l2_weight', 0, 'consistent_weight')


# Train Configuration
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 32, 'Size of each mini-batch.')
flags.DEFINE_integer('layers', 2, 'number of DeepFM Deep Component')
flags.DEFINE_string('optimizer', 'sgd', 'sgd | adam.')
flags.DEFINE_integer('steps_check',100,"")
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate. Adam: 0.001 | 0.0001')
flags.DEFINE_float("clip",5,"Gradient clip")
flags.DEFINE_string('decay_scheme', '', """ How we decay learning rate. Options include:
                    luong234: after 2/3 num train steps, we start halving the learning rate for 4 times before finishing.
                    luong5: after 1/2 num train steps, we start halving the learning rate for 5 times before finishing.
                    luong10: after 1/2 num train steps, we start halving the learning rate for 10 times before finishing.
                    """)

# Initializer configuration
flags.DEFINE_string('init_op', 'uniform', "uniform | glorot_normal | glorot_uniform")
flags.DEFINE_float('init_weight', 0.1, 'or uniform init_op, initialize weights')

FLAGS = flags.FLAGS

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                    filename=FLAGS.log_path, filemode='a', level=logging.DEBUG,
                    datefmt='%m/%d/%y %H:%M:%S')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                              datefmt='%m/%d/%y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    logger.info("loading raw data")
    train_raw = load_raw_data(FLAGS.raw_train_dataset,FLAGS.is_filter_train,FLAGS.select_more,FLAGS.filter_order_num,FLAGS.random_shuffle_train)
    dev_raw = load_raw_data(FLAGS.raw_dev_dataset,FLAGS.is_filter_test,FLAGS.select_more,FLAGS.filter_order_num,FLAGS.random_shuffle_test)

    # create data set
    logger.info("transform dataset")
    train_data = prepare_dataset(train_raw)
    dev_data = prepare_dataset(dev_raw)

    logger.info("BatchManager Processing")
    train_manager = BatchManager(train_data, FLAGS.batch_size,FLAGS.max_length)
    dev_manager = BatchManager(dev_data, FLAGS.batch_size if len(dev_data) > FLAGS.batch_size else len(dev_data),FLAGS.max_length)

    model = Model(FLAGS)
    logger.info("train starting")
    model.train_model(FLAGS, train_manager, dev_manager)


