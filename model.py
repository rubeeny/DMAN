# -*- coding:utf-8 -*-
import logging
import math
import os
import numpy as np
import itertools

import tensorflow as tf
from layers import multihead_attention, ff, dense_connect, customized_loss, get_shape_list

logger = logging.getLogger(__name__)

def calculate_rouge_s(s1,s2):
    correct_cnt = 0
    if len(s1)==0 or len(s2)==0 or len(s1)!=len(s2):
        return -99999
    pred_rank_list = list(s1)
    true_rank_list = list(s2)
    pred_combine_list = list(itertools.combinations(pred_rank_list,2))
    true_combine_list = list(itertools.combinations(true_rank_list,2))
    total_cnt = len(true_combine_list)
    for tup in pred_combine_list:
        if tup in true_combine_list:
            correct_cnt += 1
    if total_cnt == 0:
        return -99999
    correct_rate = correct_cnt*1.0 / total_cnt
    return correct_rate


def calculate_rouge_l(s1, s2):
    if len(s1)==0 or len(s2)==0 or len(s1)!=len(s2):
        return -99999
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 1
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 2
            else:
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 3
    (p1, p2) = (len(s1), len(s2))
    s = []
    while m[p1][p2]:
        c = d[p1][p2]
        if c == 1:
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 2:
            p2 -= 1
        if c == 3:
            p1 -= 1
    s.reverse()
    return len(s)*1.0/len(s1)

def get_distance(lat1, lon1, lat2, lon2):
    """
    Distance Util
    """
    d_lat1 = lat1 * math.pi / 180
    d_lat2 = lat2 * math.pi / 180
    a = d_lat1 - d_lat2
    b = lon1 * math.pi / 180 - lon2 * math.pi / 180
    s = 2 * tf.sin(
        tf.sqrt(
            tf.pow(tf.sin(a / 2), 2) + tf.cos(d_lat1) * tf.cos(d_lat2) * tf.pow(tf.sin(b / 2), 2)))
    return s * 6378137

def get_distance_e(lat1, lon1, lat2, lon2):
    """
    Distance Util
    """
    d_lat1 = lat1 * math.pi / 180
    d_lat2 = lat2 * math.pi / 180
    a = d_lat1 - d_lat2
    b = lon1 * math.pi / 180 - lon2 * math.pi / 180
    s = 2 * math.sin(
        math.sqrt(
            math.pow(math.sin(a / 2), 2) + math.cos(d_lat1) * math.cos(d_lat2) * math.pow(math.sin(b / 2), 2)))
    return s * 6378137

def reasoning_result(length, probs_):
    """
    根据业务规则，提取答案
    :param length: 实际长度
    :param probs_: 概率
    :return: 顺序
    """
    results=[]
    for i, p in enumerate(probs_):
        p=p[0:length[i],0:length[i]]#取出有效概率长度
        result=[]
        for step in range(length[i]):
            step_number=np.argmax(p[step,:])
            if step_number not in result:
                result.append(step_number)
            else:
                for  j in result:
                    p[step,j]=0
                step_number=np.argmax(p[step,:])
                result.append(step_number)
        results.append(result)
    return results

def sort_eta(pre_eta_, length):
    pred_order_etas=[]
    for eta,len_num in zip(pre_eta_,length):
        dic_eta=dict()
        eta=eta[:len_num]
        for i,e in enumerate(eta):
            dic_eta[i]=e
        pred_order_eta= [tup[0] for tup in sorted(dic_eta.items(), key=lambda d: d[1])]
        pred_order_eta_final = [pred_order_eta.index(j) for j in range(len_num)]
        pred_order_etas.append(pred_order_eta_final)
    return  pred_order_etas


def get_distance_weight(rank_labels, distance_labels,max_length,batch_size):
    rank_r = tf.reshape(rank_labels, [-1])
    dis_label_r = tf.reshape(distance_labels, [-1])
    dims = tf.unstack(tf.shape(rank_labels))

    num_batch = dims[0]
    max_len = max_length

    b_step = tf.range(0, num_batch * max_len, max_len)
    b_step = tf.expand_dims(b_step, -1)
    b_step = tf.tile(b_step, [1, max_len])
    b_step = tf.reshape(b_step, [-1])
    rank_r = tf.add(rank_r, b_step)
    weight = tf.gather(dis_label_r, rank_r)
    weight = tf.reshape(weight, [-1, max_len])

    return  weight


def compute_node_distance(elems):
    #elems = rank_lat_lngs,pred_rank_lat_lngs
    rank_lat_lng, pred_rank_lat_lng=elems
    distance = get_distance(rank_lat_lng[0],rank_lat_lng[1],pred_rank_lat_lng[0],pred_rank_lat_lng[1])

    distance = tf.where(tf.less_equal(distance,0.0),0.0,distance)
    distance = tf.where(tf.greater_equal(distance,5000.0),5000.0,distance)
    # if tf.less_equal(distance,0.0) is not  None:
    #     distance = 0.0
    # if tf.greater_equal(distance,5000.0) is not None:
    #     distance =5000.0

    distance = tf.sqrt(tf.maximum(distance,300.0)/300.0)

    return distance


def get_order2order_weight(rank_labels, pred_ranks, lat_lng_labels, max_length, batch_size):
    rank_f = tf.reshape(rank_labels, [-1])
    pred_rank_f = tf.reshape(pred_ranks, [-1])
    lat_lng_labels_f = tf.reshape(lat_lng_labels,[-1,2])

    dims = tf.unstack(tf.shape(rank_labels))
    num_batch = dims[0]
    max_len = max_length

    b_step = tf.range(0, num_batch * max_len, max_len)
    b_step = tf.expand_dims(b_step, -1)
    b_step = tf.tile(b_step, [1, max_len])
    b_step = tf.reshape(b_step, [-1])

    rank_f = tf.add(rank_f, b_step)
    pred_rank_f = tf.add(pred_rank_f,b_step)


    rank_lat_lngs = tf.gather(lat_lng_labels_f, rank_f)#(bath_size*maxlen) * 2
    pred_rank_lat_lngs =  tf.gather(lat_lng_labels_f, pred_rank_f)

    elems= rank_lat_lngs,pred_rank_lat_lngs
    weight = tf.map_fn(compute_node_distance,elems,dtype=tf.float32)
    weight = tf.reshape(weight, [-1, max_len])

    return weight

def compute_pred_eta_rank(elems):
    pre_eta_, length = elems
    pre_eta_f =tf.squeeze(pre_eta_,[-1])
    dims = tf.unstack(tf.shape(pre_eta_f))
    max_length = dims[0]
    eta = pre_eta_f[:length]
    size = tf.size(eta)
    min_index = tf.nn.top_k(-eta,size)[1]
    pre_rank = tf.nn.top_k(-min_index,size)[1]
    pred_order_eta=tf.pad(pre_rank, [[0, max_length-length]], "CONSTANT")

    return pred_order_eta

def sort_eta_train(pre_etas, enc_seq_length):
    elems = pre_etas,enc_seq_length
    pred_eta_rank = tf.map_fn(compute_pred_eta_rank, elems, dtype=tf.int32)
    return pred_eta_rank


def calculate_weigth_rank_score(true_rank_list, pred_rank_list, lat_lng_list):
    result=1
    if len(true_rank_list)==1:
        return  1
    for true_rank,pred_rank in zip(true_rank_list,pred_rank_list):
        if true_rank != pred_rank:
            lat_lng_label = lat_lng_list[true_rank]
            lat_lng_pred = lat_lng_list[pred_rank]
            distance = get_distance_e(lat_lng_label[0], lat_lng_label[1], lat_lng_pred[0],lat_lng_pred[1])
            if distance>300:
                result=0
                return  result
    return result

def cal_spearman_rho(true_rank_list, pred_rank_list):
    total=0
    n= len(true_rank_list)
    for i in range(n):
        total += (true_rank_list[i] - pred_rank_list[i]) ** 2
    spearman = 1.0 if n==1 else 1 - float(6 * total) / (n * (n ** 2 - 1))
    return spearman

def cal_kendall_tau(true_rank_list , pred_rank_list):
    length = len(true_rank_list)
    if length != len(pred_rank_list):
        return -1
    if length==1:
        return 1
    set_1 = set()
    set_2 = set()
    for i in range(length):
        for j in range(i+1,length):
            set_1.add( (true_rank_list[i],true_rank_list[j]) )
            set_2.add( (pred_rank_list[i],pred_rank_list[j]) )
    count = len(set_1 & set_2)
    return float(count)*2 / ((length-1)*length)


class Model(object):
    def __init__(self, config):
        self.global_step = tf.Variable(0, trainable=False)
        self.config=config
        self.best = tf.Variable(config.init_best, trainable=False,dtype=tf.float32)
        self.create_placeholders()
        self.enc_seq_length =tf.cast(tf.reduce_sum(tf.sign(tf.reduce_sum(tf.abs(self.inputs), axis=-1)),axis=-1),  tf.int32)
        self.enc_seq_length_op =tf.add(self.enc_seq_length,0,name="seq_length")

        self.mask_eta = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.inputs), axis=-1)), -1)  #(N, T_q,1)
        self.mask = tf.tile(self.mask_eta, [1, 1, tf.shape(self.inputs)[1]]) ##(N, T_q,T_q) # Pad为0的position不attend
        self.mask_ = self.mask * tf.transpose(self.mask, [0, 2, 1])

        def encode(inputs,rider_inputs,mask_, deep_keep_prob,isTrain):
            '''
            Returns
            memory: encoder outputs. (N, T1, d_model)
            '''
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                #we don't need embedding_lookup
                # we don't need embedding_lookup
                inputs = tf.concat([inputs, rider_inputs], axis=-1)
                enc = dense_connect(name="inputs_project", input=inputs, out_dim=config.d_model,
                                    l2_scale=config.l2_scale)
                enc *= config.d_model ** 0.5  # scale
                ## Blocks
                for i in range(config.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                        # self-attentionc
                        enc,attention,attention_raw = multihead_attention(queries=enc,
                                                  keys=enc,
                                                  values=enc,
                                                  mask_=mask_,
                                                  num_heads=config.num_heads,
                                                  dropout_rate=deep_keep_prob,
                                                  l2_scale=config.l2_scale,
                                                  train=isTrain,
                                                  causality=False)
                        # feed forward
                        enc = ff(enc, num_units=[config.d_ff, config.d_model],l2_scale=config.l2_scale)
            memory = enc
            return memory,attention,attention_raw

        def decode(final_hidden,mask):
            padding_num = -2 ** 32 + 1
            final_hidden_shape = get_shape_list(final_hidden, expected_rank=3)
            batch_size = final_hidden_shape[0]
            seq_length = final_hidden_shape[1]
            hidden_size = final_hidden_shape[2]

            output_weights = tf.get_variable(
                "output_weights", [seq_length, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [seq_length], initializer=tf.zeros_initializer())

            final_hidden_matrix = tf.reshape(final_hidden,
                                             [batch_size * seq_length, hidden_size])
            logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            logits = tf.reshape(logits, [batch_size, seq_length, seq_length])

            attention_raw = tf.matmul(logits, tf.transpose(logits, [0, 2, 1]))  # (N, T_q, T_k)
            # attention_raw = tf.matmul(final_hidden, tf.transpose(final_hidden, [0, 2, 1]))  # (N, T_q, T_k)

            mask_ =mask*tf.transpose(mask,[0,2,1])
            paddings = tf.ones_like(mask_)*padding_num

            attention_raw = tf.where(tf.equal(mask_, 0), paddings, attention_raw)
            attention_final = tf.nn.softmax(attention_raw)

            # logists_final = tf.matmul(attention_final,logits)

            return logits,attention_final

        def evalate(memory,rider_inputs,mask_,deep_keep_prob,isTrain):

            with tf.variable_scope("num_blocks_{}".format("evalate"), reuse=tf.AUTO_REUSE):
                # self-attentionc
                enc_rider = rider_inputs
                enc_rider, attention, attention_raw = multihead_attention(queries=enc_rider,
                                                                    keys=memory,
                                                                    values=memory,
                                                                    mask_=mask_,
                                                                    num_heads=config.num_heads,
                                                                    dropout_rate=deep_keep_prob,
                                                                    l2_scale=config.l2_scale,
                                                                    train=isTrain,
                                                                    causality=False)
                # feed forward
                # enc = dense_connect("evaluate", enc_rider, config.max_length, l2_scale=config.l2_scale)
                enc = tf.reduce_mean(attention,axis=1)
                enc=tf.expand_dims(enc,axis=-1)
                enc=enc*memory
                return enc

        if not config.is_use_eta_attention_layer:
            self.memory = tf.concat([self.inputs, self.rider_inputs], axis=-1)
        else:
            self.memory_, self.attention_orders, self.attention_raw = encode(self.inputs, self.rider_inputs, self.mask_,
                                                                             self.deep_keep_prob,
                                                                             self.isTrain)  # batch_size * seq_len*hidden_size
            if config.is_use_evaluation_layer:
                rider_inputs = dense_connect(name="rider_inputs", input=self.rider_inputs, out_dim=config.d_model,l2_scale=config.l2_scale)
                self.memory = evalate(self.memory_,rider_inputs,self.mask_,self.deep_keep_prob,self.isTrain)
            else:
                self.memory = self.memory_

        with tf.name_scope("rank_layer"):
            self.logits,self.attention_final= decode(self.memory,self.mask)  # batch_size*seq_length*seq_lengt
            self.probs_ = tf.nn.softmax(self.logits,axis=1)
            self.probs = tf.clip_by_value(self.probs_, 1e-10, 1.0)*self.mask_

            self.pred_rank = tf.argmax(self.probs, axis=-1, name="predictions_class",output_type=tf.int32)
            # self.pred_score = tf.reduce_max(self.probs, axis=-1, name="predictions_score")

            # mask = self.mask * tf.transpose(self.mask, [0, 2, 1])
            # paddings = tf.ones_like(self.mask) * padding_num
            # self.logits_ = tf.where(tf.equal(self.logits, 0), paddings, self.logits)

            # paddings = tf.ones_like(self.mask) * padding_num
            # : 0.680672268908, online_ape20:0.59243697479
            # self.logits_ = tf.where(tf.equal(mask_, 0), paddings, self.probs_)
            # self.probs = tf.log(self.logits_)

        with tf.name_scope("eta_layer"):
            self.pre_eta_d = tf.nn.dropout(self.memory, self.deep_keep_prob)
            self.pre_eta_f1 = dense_connect('predict', self.pre_eta_d
                                            , config.d_model, None, l2_scale=config.l2_scale)  # batch_size * seq_len*d
            # self.pre_eta_s = tf.contrib.layers.layer_norm(self.pre_eta_f1)
            self.pre_eta_c = tf.concat([self.pre_eta_f1, self.logits], axis=-1)
            if config.use_multi_task:
                self.pre_eta_ = dense_connect('predict1', self.pre_eta_c, 1, None,
                                              l2_scale=config.l2_scale)  # batch_size * seq_len*1
            else:
                self.pre_eta_ = dense_connect('predict2', self.pre_eta_f1, 1, None,
                                              l2_scale=config.l2_scale)  # batch_size * seq_len*1

            self.pre_eta_op = tf.squeeze(self.pre_eta_, axis=-1)

            # self.pre_eta_ =tf.layers.dense(self.memory,1,use_bias=None)
            self.pre_eta_op = tf.add(self.pre_eta_op, 0, name="predict_eta_op")

        # #计算损失
        self.loss_len = tf.cast(self.enc_seq_length, dtype=tf.float32) + 1

        self.loss_eta = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(self.labels-self.pre_eta_),axis=-1),axis=-1)/self.loss_len)
        # self.loss_eta=tf.losses.mean_squared_error(self.labels,self.pre_eta_*self.mask_eta)
        # #labels batch_size*seq_length
        # self.loss_eta=customized_loss(self.labels,self.pre_eta_,config.threshold_1,config.threshold_2,config.penalty_1,config.penalty_2)*self.mask_eta
        #行one_hot
        self.one_hot_labels = tf.one_hot(self.rank_labels,depth=config.max_length,dtype=tf.float32)*self.mask_
        # true_dis_weight = get_distance_weight(self.rank_labels,self.distance_labels,config.max_length,config.batch_size)
        # pred_dis_weight = get_distance_weight(self.pred_rank,self.distance_labels,config.max_length,config.batch_size)
        # # weight = tf.sqrt(tf.abs(true_dis_weight-pred_dis_weight))
        # if config.is_use_weight:
        #     self.loss_rank_tmp = tf.reduce_sum(
        #         tf.reduce_sum(tf.log(self.probs_) * self.one_hot_labels, axis=-1) * self.weight, axis=-1)
        # else:
        self.loss_rank_tmp = tf.reduce_sum(tf.reduce_sum(tf.log(self.probs_) * self.one_hot_labels, axis=-1),
                                               axis=-1)
        self.loss_rank= -tf.reduce_mean(self.loss_rank_tmp/self.loss_len)
        self.pred_rank_etas = sort_eta_train(self.pre_eta_, self.enc_seq_length)
        self.one_hot_pred_eta_rank_labels = tf.one_hot(self.pred_rank_etas, depth=config.max_length,
                                                       dtype=tf.float32)*self.mask_
        if config.is_use_weight:
            weight = get_order2order_weight(self.rank_labels, self.pred_rank, self.lat_lng_labels,config.max_length,config.batch_size)
            self.weight = weight * tf.squeeze(self.mask_eta)
            self.log_loss_pred_rank_eta_2_pred_pointer_rank = tf.reduce_sum(tf.reduce_sum(tf.log(self.probs_) * self.one_hot_pred_eta_rank_labels, axis=-1)*self.weight,
                                               axis=-1)
        else:
            # AB = tf.reduce_sum(self.one_hot_labels * self.one_hot_pred_eta_rank_labels, axis=-1)
            # sqrtA = tf.sqrt(tf.reduce_mean(tf.square(self.one_hot_labels),axis=-1))
            # sqrtB = tf.sqrt(tf.reduce_mean(tf.square(self.one_hot_pred_eta_rank_labels),axis=-1))
            # self.log_loss_pred_rank_eta_2_pred_pointer_rank = tf.cos(AB/(sqrtA*sqrtB))
            # self.log_loss_pred_rank_eta_2_pred_pointer_rank = tf.reduce_sum(self.log_loss_pred_rank_eta_2_pred_pointer_rank* tf.squeeze(self.mask_eta),
            #     axis=-1)
            self.deta = tf.reduce_sum(tf.abs(self.one_hot_labels - self.one_hot_pred_eta_rank_labels),axis=-1)
            self.log_loss_pred_rank_eta_2_pred_pointer_rank = tf.reduce_sum(self.deta,axis=-1)
            # self.log_loss_pred_rank_eta_2_pred_pointer_rank = tf.reduce_sum(tf.reduce_sum(tf.log(self.probs_) * self.one_hot_pred_eta_rank_labels, axis=-1),
            #                                    axis=-1)
        self.log_loss_rank = tf.reduce_mean(self.log_loss_pred_rank_eta_2_pred_pointer_rank/self.loss_len)

        self.loss_l2=tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        if config.use_multi_task:
            self.loss = config.main_task_weight*self.loss_eta+config.rank_task_weight*self.loss_rank+ config.l2_weight *self.loss_l2
            if config.is_use_log_loss:
                self.loss += config.consistent_weight * self.log_loss_rank
            else:
                self.loss += 0 * self.log_loss_rank
            logger.info("self.loss_eta+100*self.loss_rank+0.0*self.loss_l2")
        else:
            self.loss = self.loss_eta+0.0*self.loss_rank+0.0*self.loss_l2+0.0*self.log_loss_rank
            logger.info("self.loss_eta")
        self.opt = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        grads_and_vars = self.opt.compute_gradients(self.loss)
        capped_grads_vars = [[tf.clip_by_value(g, -config.clip, config.clip), v]
                             for g, v in grads_and_vars]

        self.train_op = self.opt.apply_gradients(capped_grads_vars, global_step=self.global_step)

    def create_feed_dict(self, batch, isTrain=True):
        """
        Create the dictionary of data to feed to tf session during training.
        """
        feed_dict = {
            self.labels: batch[0],
            self.rank_labels: batch[2],
            self.inputs: batch[4],
            self.rider_inputs:batch[5],
            self.distance_labels:batch[6],
            self.lat_lng_labels:batch[7],
            self.deep_keep_prob: self.config.dropout_rate if isTrain else 1.0,
            self.isTrain: isTrain
        }
        return feed_dict

    def create_placeholders(self):
        #输入,我们这里输入为三维向量，不需要look_up embeding 最后一维是特征的个数 batch_size*seqlen*feat_size
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.config.max_length, 127], name="inputs")
        self.rider_inputs = tf.placeholder(tf.float32, shape=[None, self.config.max_length, 22], name="rider_inputs")
        self.labels = tf.placeholder(tf.float32, shape=[None, self.config.max_length,1], name="label")#batch_size*seqlen
        self.rank_labels = tf.placeholder(tf.int32, shape=[None, self.config.max_length], name="order_label")#batch_size*seqlen
        self.distance_labels = tf.placeholder(tf.float32, shape=[None, self.config.max_length], name="distance_labels")#batch_size*seqlen
        self.lat_lng_labels = tf.placeholder(tf.float32, shape=[None, self.config.max_length,2], name="lat_lng_labels")#batch_size*seqlen
        self.deep_keep_prob = tf.placeholder(tf.float32, name="deep_keep_prob")
        self.isTrain = tf.placeholder(tf.bool, name="isTrain")


    def train_model(self, FLAGS, train_manager, dev_manager):
        """Train the model.
        """
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        steps_per_epoch = train_manager.len_data
        # 启动会话，执行整个任务
        with tf.Session(config=tf_config) as sess:
            if FLAGS.isrestore and  tf.train.checkpoint_exists(FLAGS.save_parameters):
                # Fit the model
                logger.info("##########Start from trained###########")
                self.tf_saver = tf.train.Saver()
                self.tf_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.save_parameters))
            else:
                # Fit the model
                logger.info("##########Start training###########")
                # Initialize tf stuff
                summary_objs = self.init_tf_ops(sess)
                self.tf_merged_summaries = summary_objs[0]
                self.tf_summary_writer = summary_objs[1]
                self.tf_saver = summary_objs[2]
            for i in range(400):
                for batch in train_manager.iter_batch(shuffle=True):
                    feed_dict = self.create_feed_dict(batch,isTrain=True)
                    step, batch_loss,loss_rank,loss_eta,log_loss_rank = self.fit(sess, feed_dict, batch)
                    if step % FLAGS.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("iteration:{} step:{}/{},train loss:{:>9.6f},loss_rank:{:>9.6f},loss_eta:{:>9.6f},log_loss_rank:{:>9.6f},".format(
                            iteration, step % steps_per_epoch, steps_per_epoch, np.mean(batch_loss),np.mean(loss_rank),np.mean(loss_eta),np.mean(log_loss_rank)))
                    if step % (FLAGS.steps_check*1) == 0:
                        # whether to be best
                        _, best = self.evaluate(sess, dev_manager)
                        if best:
                            # Save the model paramenters
                            if FLAGS.save_parameters:
                                self.tf_saver.save(sess, os.path.join(FLAGS.save_parameters, "model"))
                                logger.info("model saved")

    def fit(self, sess, feed_dict, batch):
        """Fit the model to the data.
        Parameters
         ----------
        sess : Tensorflow Session.
        batch : batch labels,text_feats, stat_feats
        feed_dict:
        """
        assert len(batch[0]) == len(batch[1]) == len(batch[2])
        # Train model
        try:
            log_loss_rank,loss_train,loss_rank,loss_eta, step,length ,_ = sess.run([self.log_loss_rank,self.loss,self.loss_rank,self.loss_eta, self.global_step, self.enc_seq_length,self.train_op], feed_dict=feed_dict)
        except Exception as e:
            print(e)
            return -999999, -999999

        return step, loss_train ,loss_rank,loss_eta,log_loss_rank

    def init_tf_ops(self,sess):
        """
        Initialize TensorFlow operations.
        """
        summary_merged = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init_op)
        # tensorboard_dir = '/home/longzhangchao/data/model/etaOrderModel/tensorboard'
        tensorboard_dir = self.config.tensorboard_dir
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

        return summary_merged, summary_writer, saver

    def evaluate(self, sess, batch_manager):
        """
        evaluate the model over the eval set.
        """
        total_loss = 0.0
        zero_cnt = 0
        correct_10 = 0
        correct_o_10 = 0
        ape_20_cnt = 0
        ape_o_20_cnt = 0
        sum_square = 0
        o_sum_square = 0
        sum_abs_err = 0
        o_sum_abs_err = 0
        sum_abs = 0
        o_sum_abs = 0
        total_order=0

        total_cnt_rank=0
        pred_rouge_s=0
        pred_rouge_l=0
        online_rouge_s=0
        online_rouge_l=0
        correct_rank=0
        online_correct_rank=0
        log_cnt_rank=0
        eta_correct_rank = 0

        eta_sum_weight_rank_score = 0
        o_sum_weight_rank_score = 0
        sum_weight_rank_score=0

        eta_sum_spearman_rho_score = 0
        o_sum_spearman_rho_score = 0
        sum_spearman_rho_score=0

        eta_sum_kendall_tau_score = 0
        o_sum_kendall_tau_score = 0
        sum_kendall_tau_score = 0



        for batch in batch_manager.iter_batch(shuffle=False):
            try:
                batch_len = len(batch[1])
                total_cnt_rank += batch_len
                feed_dict = self.create_feed_dict(batch,isTrain=False)
                length,pre_eta_,probs,pred_order_etas= sess.run([self.enc_seq_length,self.pre_eta_,self.probs,self.pred_rank_etas], feed_dict=feed_dict)
                # length,pre_eta_,logits,attention_final,one_hot_labels,probs,pred_rank_p ,pred_order_etas= sess.run([self.enc_seq_length,self.pre_eta_,self.logits,self.attention_final,self.one_hot_labels,self.probs,self.pred_rank,self.pred_rank_etas], feed_dict=feed_dict)
                # pred_order_etas=sort_eta(pre_eta_,length)
                pred_ranks=reasoning_result(length,probs)
                # 排序准确率
                for pred_rank,pred_order_eta,true_rank,online_rank,true_length,lat_lng_label in zip(np.array(pred_ranks),np.array(pred_order_etas),np.array(batch[2]), np.array(batch[3]),np.array(length).flatten(),np.array(batch[7])):
                    if true_length ==0:
                        total_cnt_rank -=1
                        continue
                    true_rank_str = "".join([str(i) for i in true_rank[0:true_length]])
                    pred_rank_str = "".join([str(i) for i in pred_rank[0:true_length]])
                    pred_order_eta_str = "".join([str(i) for i in pred_order_eta[0:true_length]])
                    online_rank_str = "".join([str(i) for i in online_rank[0:true_length]])

                    eta_sum_weight_rank_score += calculate_weigth_rank_score(true_rank[0:true_length],pred_order_eta[0:true_length],lat_lng_label[0:true_length])
                    o_sum_weight_rank_score += calculate_weigth_rank_score(true_rank[0:true_length],online_rank[0:true_length],lat_lng_label[0:true_length])
                    sum_weight_rank_score+= calculate_weigth_rank_score(true_rank[0:true_length],pred_rank[0:true_length],lat_lng_label[0:true_length])

                    eta_sum_spearman_rho_score += cal_spearman_rho(true_rank[0:true_length],pred_order_eta[0:true_length])
                    o_sum_spearman_rho_score += cal_spearman_rho(true_rank[0:true_length],online_rank[0:true_length])
                    sum_spearman_rho_score += cal_spearman_rho(true_rank[0:true_length],pred_rank[0:true_length])

                    eta_sum_kendall_tau_score += cal_kendall_tau(true_rank[0:true_length],pred_order_eta[0:true_length])
                    o_sum_kendall_tau_score += cal_kendall_tau(true_rank[0:true_length],online_rank[0:true_length])
                    sum_kendall_tau_score += cal_kendall_tau(true_rank[0:true_length],pred_rank[0:true_length])


                    # if calculate_rouge_s(pred_order_eta_str,true_rank_str) == -99999 or calculate_rouge_l(pred_order_eta_str,true_rank_str) == -99999:
                    #     total_cnt_rank -=1
                    #     continue
                    # pred_rouge_s+=calculate_rouge_s(pred_rank_str,true_rank_str)
                    # pred_rouge_l+=calculate_rouge_l(pred_rank_str,true_rank_str)
                    # online_rouge_s+=calculate_rouge_s(online_rank_str,true_rank_str)
                    # online_rouge_l+=calculate_rouge_l(online_rank_str,true_rank_str)

                    if true_rank_str == pred_rank_str:
                        correct_rank += 1
                    if true_rank_str == online_rank_str:
                        online_correct_rank+=1
                    if true_rank_str == pred_order_eta_str:
                        eta_correct_rank +=1
                    log_cnt_rank += 1
                    if log_cnt_rank % 5000 == 0:
                        logger.info("prediction is:{},true is {},online_rank is {},eta_rank is {}".format(pred_rank_str, true_rank_str,online_rank_str,pred_order_eta_str))


                for true_label,online_pre,pre_label,true_l in zip(np.array(batch[0]),np.array(batch[1]),np.array(pre_eta_),length):
                    if true_l == 0:
                        total_cnt_rank -= 1
                        continue
                    true_label = true_label[0:true_l,:]
                    pre_label = pre_label[0:true_l,:]
                    online_pre = online_pre[0:true_l,:]
                    for t_l,o_l,p_l in zip(true_label,online_pre,pre_label):
                        if t_l[0] == 0:
                            zero_cnt += 1
                            total_order -=1
                            continue
                        total_order += 1
                        sum_square+=(t_l[0]-p_l[0])**2
                        o_sum_square+=(t_l[0]-o_l[0])**2
                        sum_abs_err+=np.abs(t_l[0]-p_l[0])/t_l[0]
                        o_sum_abs_err+=np.abs(t_l[0]-o_l[0])/t_l[0]
                        sum_abs += np.abs(t_l[0] - p_l[0])
                        o_sum_abs += np.abs(t_l[0] - o_l[0])
                        if np.abs(t_l[0]-p_l[0])<10:
                            correct_10+=1
                        if np.abs(t_l[0]-o_l[0])<10:
                            correct_o_10+=1
                        if np.abs(t_l[0]-p_l[0])/ t_l[0] < 0.3:
                            ape_20_cnt+=1
                        if np.abs(t_l[0]-o_l[0])/ t_l[0] < 0.3:
                            ape_o_20_cnt+=1
                        if total_order%5000==0:
                            logger.info("true eta is {},model_pred_eta is {},online_pred_eta is {}".format(t_l[0],p_l[0],o_l[0]))
            except Exception as e:

                logger.info(e.message)
                logger.info("evalate error pass")
                pass

        acc_10= 0 if total_order==0 else correct_10*1.0/total_order
        acc_O_10= 0 if total_order==0 else correct_o_10*1.0/total_order
        ape_20 = 0 if total_order==0 else ape_20_cnt*1.0/total_order
        ape_o_20 = 0 if total_order==0 else ape_o_20_cnt*1.0/total_order

        mse = 0 if total_order==0 else sum_square*1.0/total_order
        o_mse = 0 if total_order==0 else o_sum_square *1.0/total_order
        mae = 0 if total_order==0 else sum_abs*1.0/total_order
        o_mae = 0 if total_order==0 else o_sum_abs*1.0/total_order
        mape = 0 if total_order==0 else sum_abs_err*1.0/total_order
        o_mape = 0 if total_order==0 else o_sum_abs_err*1.0/total_order


        acc_online_rank = 0 if total_cnt_rank==0 else online_correct_rank*1.0/total_cnt_rank
        acc_pred_rank = 0 if total_cnt_rank==0 else correct_rank*1.0/total_cnt_rank
        p_rouge_s = 0 if total_cnt_rank==0 else pred_rouge_s*1.0/total_cnt_rank
        p_rouge_l = 0 if total_cnt_rank==0 else pred_rouge_l*1.0/total_cnt_rank
        o_rouge_s = 0 if total_cnt_rank==0 else online_rouge_s*1.0/total_cnt_rank
        o_rouge_l = 0 if total_cnt_rank==0 else online_rouge_l*1.0/total_cnt_rank
        acc_eta_pred_rank = 0 if total_cnt_rank == 0 else eta_correct_rank * 1.0 / total_cnt_rank

        eta_kendall_tau_score = 0 if total_cnt_rank==0 else eta_sum_kendall_tau_score*1.0/total_cnt_rank
        o_kendall_tau_score = 0 if total_cnt_rank==0 else o_sum_kendall_tau_score*1.0/total_cnt_rank
        kendall_tau_score = 0 if total_cnt_rank==0 else sum_kendall_tau_score*1.0/total_cnt_rank

        eta_spearman_rho_score = 0 if total_cnt_rank==0 else eta_sum_spearman_rho_score*1.0/total_cnt_rank
        o_spearman_rho_score = 0 if total_cnt_rank==0 else o_sum_spearman_rho_score*1.0/total_cnt_rank
        spearman_rho_score = 0 if total_cnt_rank==0 else sum_spearman_rho_score*1.0/total_cnt_rank

        eta_weight_rank_score =  0 if total_cnt_rank==0 else eta_sum_weight_rank_score*1.0/total_cnt_rank
        o_weight_rank_score =  0 if total_cnt_rank==0 else o_sum_weight_rank_score*1.0/total_cnt_rank
        weight_rank_score =  0 if total_cnt_rank==0 else sum_weight_rank_score*1.0/total_cnt_rank


        best = self.best.eval()
        is_best = False
        if mae<best:
            tf.assign(self.best, mae).eval()
            logger.info("best acc_10 : {},best acc_O_10 : {} ".format(acc_10,acc_O_10))
            logger.info("best p_rouge_s : {},best o_rouge_s : {} ".format(p_rouge_s,o_rouge_s))
            logger.info("best p_rouge_l : {},best o_rouge_l : {} ".format(p_rouge_l,o_rouge_l))
            logger.info("best kendall_tau_score:{},best eta_kendall_tau_score:{},best o_kendall_tau_score:{} ".format(kendall_tau_score,eta_kendall_tau_score,o_kendall_tau_score))
            logger.info("best spearman_rho_score:{},best eta_spearman_rho_score:{},best o_spearman_rho_score:{}".format(spearman_rho_score,eta_spearman_rho_score,o_spearman_rho_score))
            logger.info("best weight_rank_score:{},best eta_weight_rank_score:{},best o_weight_rank_score:{} ".format(weight_rank_score,eta_weight_rank_score,o_weight_rank_score))
            logger.info("best acc_pred_rank : {}, acc_online_rank:{},acc_eta_pred_rank:{}".format(acc_pred_rank,acc_online_rank,acc_eta_pred_rank))
            logger.info("best pred_ape20 : {}, online_ape20:{} ".format(ape_20,ape_o_20))
            logger.info("best pred_mse : {}, online_mse:{}, ".format(mse,o_mse))
            logger.info("best pred_mae : {}, online_mae:{}, ".format(mae, o_mae))
            logger.info("best pred_mape : {}, online_mape:{}, ".format(mape,o_mape))
            logger.info("zero_cnt: {}".format(zero_cnt))

            is_best = True
        return total_loss, is_best









