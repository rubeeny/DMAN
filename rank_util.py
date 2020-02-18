# -*- coding:utf-8 -*-
"""
Data Process Util
"""
import time
import math
import json
import pickle
import logging
import numpy as np
import random
from text_util import process_text,text_split
logger = logging.getLogger('utilities')

weather_condition_dic = {"小雨":1,"晴":2,"多云":3,"少云":4,"中雨":5,"雾":6,"大雨":7,"霾":8,"小雪":9,
                         "中雪":10,"大雪":11}

eleme_sc_dict = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,11:7,13:8,45:9}

def get_distance(lat1, lon1, lat2, lon2):
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

def edit_distance(s1, s2):

    len_str1 = len(s1) + 1
    len_str2 = len(s2) + 1

    matrix = [[0] * (len_str2) for i in range(len_str1)]

    for i in range(len_str1):
        for j in range(len_str2):
            if i == 0 and j == 0:
                matrix[i][j] = 0
            # 初始化矩阵
            elif i == 0 and j > 0:
                matrix[0][j] = j
            elif i > 0 and j == 0:
                matrix[i][0] = i
            # flag
            elif s1[i - 1] == s2[j - 1]:
                matrix[i][j] = min(matrix[i - 1][j - 1], matrix[i][j - 1] + 1, matrix[i - 1][j] + 1)
            else:
                matrix[i][j] = min(matrix[i - 1][j - 1] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j] + 1)
    return matrix[len_str1 - 1][len_str2 - 1]



def protect_speed(speed_str, type):
    if speed_str is None:
        return 250 if type == 1 else 200
    if speed_str.lower() != "null":
        if float(speed_str) == 0:
            return 250 if type == 1 else 200
    return speed_str



def encode_discrete_feat(items):
    discrete_feat = []
    place_1h_one_hot = np.zeros([24])
    elmsc_one_hot = np.zeros([10])
    demot_one_hot = np.zeros([2])
    peek_one_hot = np.zeros([2])
    weekday_one_hot = np.zeros([7])
    iswkends_one_hot = np.zeros([2])
    diffc_one_hot = np.zeros([2])
    weather_one_hot=np.zeros([17])
    sex_one_hot = np.zeros([2])
    level_one_hot = np.zeros([5])
    fresh_one_hot = np.zeros([2])
    status_one_hot = np.zeros([2])
    food_taken_one_hot = np.zeros([2])
    items = ["null" if item == "NULL" or not item or item=="" else item for item in items]
    try:
        for j, v in enumerate(items):
            v = "0" if v == "null" or v[0] == "-" else v
            if j == 0:
                v=int(v)
                place_1h_one_hot[v] = 1
                discrete_feat.extend(place_1h_one_hot)
            if j == 1:
                v=int(v)
                elmsc_one_hot[eleme_sc_dict[v]] = 1
                discrete_feat.extend(elmsc_one_hot)
            if j == 2:
                v=int(v)
                demot_one_hot[v] = 1
                discrete_feat.extend(demot_one_hot)
            if j == 3:
                v=int(v)
                peek_one_hot[v] = 1
                discrete_feat.extend(peek_one_hot)
            if j == 4:
                v=int(v)
                weekday_one_hot[v] = 1
                discrete_feat.extend(weekday_one_hot)
            if j == 5:
                v = int(v)
                iswkends_one_hot[v]=1
                discrete_feat.extend(iswkends_one_hot)
            if j == 6:
                v = int(v)
                diffc_one_hot[v] = 1
                discrete_feat.extend(diffc_one_hot)
            if j == 7:
                v=v.strip().encode("utf-8")
                if v in weather_condition_dic:
                    weather_one_hot[weather_condition_dic[v]] = 1
                else:
                    weather_one_hot[0] = 1
                discrete_feat.extend(weather_one_hot)
            if j == 8:
                v = int(v)
                sex_one_hot[v] = 1
                discrete_feat.extend(sex_one_hot)
            if j == 9:
                v = int(v)
                level_one_hot[v/5]=1           # 骑手等级1到5，数值分别对应5、10、15、20
                discrete_feat.extend(level_one_hot)
            if j == 10:
                v = int(v)
                fresh_one_hot[v] = 1
                discrete_feat.extend(fresh_one_hot)
            if j == 11:
                v = int(v)
                status_one_hot[v] = 1
                discrete_feat.extend(status_one_hot)
            if j == 12:
                v = int(v)
                food_taken_one_hot[v] = 1
                discrete_feat.extend(food_taken_one_hot)
        return discrete_feat
    except Exception as e:
        logger.info("encode_discrete_feat error!")
        logger.info(e)
        return []

def encode_continuous_feat(items):
    continuous_feat=[]
    items=[-1 if item == "NULL" or not item or item=="" else item for item in items]
    try:
        for j,v in enumerate(items):
            if j==29 or j==30:continue
            if j in [5,7,9,10,22,23,26,28,30,32,34]:
                v = min(float(v), 1)
                continuous_feat.append(v)
            elif j in [15,17,35]:
                v = min(float(v)/10, 1)
                continuous_feat.append(v)
            elif j in [16]:
                v = min(float(v)/20, 1)
                continuous_feat.append(v)
            elif j in [0,8,11,15,31]:
                v = min(float(v)/50, 1)
                continuous_feat.append(v)
            elif j in [1,2,3,12,18,25,27,29]:
                v = min(float(v)/100, 1)
                continuous_feat.append(v)
            elif j in [19,20,21]:
                v = min(float(v)/500, 1)
                continuous_feat.append(v)
            elif j in [4,6,14,24]:
                v=min(float(v)/1000,1)
                continuous_feat.append(v)
            elif j in [33,36]:
                v=min(float(v)/10000,1)
                continuous_feat.append(v)
            elif j==13:
                continuous_feat.append(0)      # 与线上保持一致，弃用该特征
        return continuous_feat
    except Exception as e:
        logger.info("encode_continuous_feat error!")
        logger.info(e.message)
        return []

def encode_additional_feat(items):
    continuous_feat=[]
    items=[-1 if item == "NULL" or not item or item=="" else item for item in items]
    try:
        for j,v in enumerate(items):
            if j in [3,4,30]:
                v = min(float(v)/10, 1)
                continuous_feat.append(v)
            elif j in [1,9,10,11,21,22,23,29]:
                v = min(float(v)/100, 1)
                continuous_feat.append(v)
            elif j in [24]:
                v=min(float(v)/1000,1)
                continuous_feat.append(v)
            elif j in [0,2,5,6,7,8,25]:
                v=min(float(v)/10000,1)
                continuous_feat.append(v)
            else:
                v = min(float(v), 1)
                continuous_feat.append(v)
        return continuous_feat
    except Exception as e:
        logger.info("encode_additional_feat error!")
        logger.info(e.message)
        return []

def load_raw_data(data_path,is_filter,select_more,filter_order_num,random_shuffle=1):
    """
    加载数据
    :param data_path: 数据路径
    :param is_filter: 是否过滤身上订单量较少的骑手
    :param select_more: 选择身上订单量大于等于filter_order_num的样本
    :param filter_order_num: 若骑手身上订单数不为filter_order_num,过滤该样本.
    :param random_shuffle: 是否随机打乱骑手身上的订单顺序
    :return: labels,discrete_feats,continuous_feats,text_feats,id_feats,road_feats
             labels:[[label1],[label2],[label3]],
             discreate_feat:[[],[],[]]
             continuous_feats:[[],[],[]]
             text_feats:[[],[],[]]
             id_feats:[[],[],[]]
    """
    num_dispatched = 0
    num_wx=0
    total=0

    with open(data_path,"r") as rf:
        rf.readline()
        line=rf.readline()
        labels = []; online_preds=[];online_labels = [];rank_labels=[]
        distance_labels=[]
        lat_lng_labels=[]
        final_feats = []; text_feats = []; id_feats = []; order_ids = []
        rider_feats = []

        while line:
            total+=1
            if total % 10000 == 0:
                logger.info("proceessed {} exsamples".format(total))
            #丢弃空行
            if line=="" or line.strip()=="":
                line=rf.readline()
                num_wx+=1
                continue
            items=line.strip().split("\t")
            items = items[0:-1]
            #丢弃维度不对的样本
            if len(items)!=9:
                logger.info("原始数据维度不对,样本丢弃")
                logger.info(items)
                num_wx+=1
                line=rf.readline()
                continue
            try:
                # 处理正常样本
                label=[]; online_label=[]; rank_label=[]
                distance_list=[]
                lat_lng_list=[]
                online_pred_list=[]; online_pred_dict={}; label_dict={}
                final_feat=[]; text_feat=[]; id_feat=[]; order_id = []
                rider_feat=[]

                feat_dict = json.loads(items[-1])
                feat_list = feat_dict["order_feats_list"]

                if is_filter:
                    if select_more:
                        if len(feat_list)<int(filter_order_num):
                            line = rf.readline()
                            continue
                    else:
                        if len(feat_list)!=int(filter_order_num):
                            line = rf.readline()
                            continue

                rider_lng=items[3]
                rider_lat=items[4]
                for feats in feat_list:
                    # label_v = min(float(feats[0])/100.0, 1)
                    label_v = min(float(feats[0]), 100.0)

                    # 新增特征
                    online_pred =  min(float(feats[1]), 100.0)
                    # online_pred = min(float(feats[1])/100.0, 1)
                    status = int(feats[-4])
                    is_require = int(feats[-3])
                    if is_require:continue
                    L_DSTm = float(feats[-2])/100.0
                    R_DSTm = float(feats[-1])/100.0

                    #id 特征(以及维度高的palce_5m)全部拼接起来，当作文本特征看待
                    id_text=process_text("+".join(feats[2:5]))
                    id_text_ltp=text_split(id_text,isId=True)
                    id_feat.append(id_text_ltp)

                    #离散型特征处理
                    discrete_feat = encode_discrete_feat(feats[5:18])
                    rider_discrete_feat=discrete_feat[-13:-4]
                    del discrete_feat[-13:-4]
                    order_discrete_feat = discrete_feat

                    #连续型特征处理，并加入线上预计的路径规划时间
                    continuous_feat = []
                    continuous_feat.append(float(feats[18])/50.0)
                    feats[39] = protect_speed(feats[39],type=1)
                    feats[40] = protect_speed(feats[40],type=0)
                    tmp_continuous_feat = encode_continuous_feat(feats[19:56])
                    tmp_continuous_feat[-1] = 0   # 弃用distance_sum
                    additional_feat=encode_additional_feat(feats[56:85])
                    tmp_continuous_feat.extend(additional_feat)
                    continuous_feat.extend(tmp_continuous_feat)
                    continuous_feat.extend([online_pred,status,is_require,0.0,0.0])
                    continuous_feat = [-1 if v<0 else v for v in continuous_feat]
                    rider_continuous_feat=continuous_feat[13:26]
                    del continuous_feat[13:26]
                    order_continuous_feat=continuous_feat

                    if len(order_discrete_feat)!=70 or len(order_continuous_feat)!=57:
                        # line=rf.readline()
                        logger.info("离散特征或连续特征处理异常")
                        logger.info(feats)
                        num_wx+=1
                        continue
                    order_discrete_feat.extend(order_continuous_feat)
                    rider_discrete_feat.extend(rider_continuous_feat)

                    final_feat.append(order_discrete_feat)
                    rider_feat.append(rider_discrete_feat)

                    addr_text=process_text(" ".join(feats[85:87]))
                    # addr_text=process_text(" ".join(feats[56:58]))
                    text_ltp=text_split(addr_text)
                    text_feat.append(text_ltp)
                    order_id.append(int(feats[-5]))
                    #获得派单时刻骑手到订单用户的距离
                    distance=get_distance(float(rider_lat), float(rider_lng), float(feats[-7]), float(feats[-8]))
                    distance_list.append(distance*1.3)
                    lat_lng_list.append([float(feats[-7]), float(feats[-8])])
                    online_pred_list.append(online_pred)
                    label.append(label_v)
                # 计算用于排序的label
                for i,v in enumerate(online_pred_list):
                    online_pred_dict[i] = v
                # 计算用于排序的label
                for i, v in enumerate(label):
                    label_dict[i] = v               # note: i代表订单编号，0代表订单A，1代表订单B
                online_label_tmp = [tup[0] for tup in sorted(online_pred_dict.items(), key=lambda d: d[1])]
                rank_label_tmp = [tup[0] for tup in sorted(label_dict.items(), key=lambda d: d[1])]

                online_label = [online_label_tmp.index(i) for i in range(len(online_label_tmp))]
                rank_label = [rank_label_tmp.index(i) for i in range(len(rank_label_tmp))]
                #随机打乱顺序
                random_rank=[i for i in range(len(online_label_tmp))]
                if random_shuffle:
                    random.shuffle(random_rank)

                labels.append([ [label_dict[i]] for i in random_rank])
                online_preds.append([ [online_pred_dict[i]] for i in random_rank])
                distance_labels.append([distance_list[i] for i in random_rank])
                lat_lng_labels.append([lat_lng_list[i] for i in random_rank])

                online_labels.append([online_label[i] for i in random_rank])
                #以前代码可能有误
                # online_labels.append(online_label)
                rank_labels.append([rank_label[e] for e in random_rank])

                final_feats.append([final_feat[i] for i in random_rank])
                rider_feats.append([rider_feat[i] for i in random_rank])

                id_feats.append([id_feat[i]  for i in random_rank])
                text_feats.append([text_feat[i]  for i in random_rank])
                order_ids.append([order_id[i]  for i in random_rank])

            except Exception as e:
                logger.info("数据处理异常")
                num_wx += 1
                logger.info(items)
                line = rf.readline()
                logger.info(e)
            if total%10000==0:
                logger.info("无效丢弃样本数：{},已派订单数:{}，有效样本数：{}".format(num_wx,num_dispatched,total))
            line = rf.readline()

        return [labels,online_preds,rank_labels,online_labels,final_feats,rider_feats,distance_labels,lat_lng_labels,text_feats,id_feats,order_ids]


def prepare_dataset(raw_data):
    """
    改变数据集的数据组织方式
    """
    dataset = []
    labels = raw_data[0]
    online_preds=raw_data[1]
    rank_labels=raw_data[2]
    online_labels = raw_data[3]
    final_feats = raw_data[4]
    rider_feats = raw_data[5]
    distance_labels =raw_data[6]
    lat_lng_labels = raw_data[7]
    order_ids = raw_data[-1]

    datas = zip(labels,online_preds,rank_labels, online_labels, final_feats,rider_feats,distance_labels,lat_lng_labels,order_ids)

    for data in datas:
        label, online_pred,rank_label, online_label, final_feat,rider_feat,distance_label,lat_lng_label,order_id = data
        dataset.append([label,online_pred,rank_label, online_label, final_feat,rider_feat,distance_label,lat_lng_label,order_id])

    return dataset

class BatchManager(object):
    """
    create bath data
    """
    def __init__(self, data, batch_size,max_length):
        self.batch_data = self.sort_and_pad(data, batch_size,max_length)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size,max_length):
        num_batch = int(math.ceil(len(data)*1.0 / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size: (i+1)*batch_size],max_length))
        return batch_data

    @staticmethod
    def pad_data(data,max_length):
        labels = []
        online_preds=[]
        rank_labels=[]
        online_labels=[]
        final_feats = []
        rider_feats = []
        distance_labels=[]
        lat_lng_labels=[]
        order_ids = []
        for line in data:
            label, online_pred,rank_label,online_label,final_feat,rider_feat,distance_label,lat_lng_label,order_id = line
            pad_length = max_length-len(label)
            for _ in range(pad_length):
                label.append([0])
                online_pred.append([0])
                online_label+=[0]
                rank_label+=[0]
                final_feat.append([0]*127)
                rider_feat.append([0]*22)
                distance_label+=[0]
                lat_lng_label+=[[0.0,0.0]]
                order_id+=[0]
            labels.append(label[0:max_length])
            online_preds.append(online_pred[0:max_length])
            rank_labels.append(rank_label[0:max_length])
            online_labels.append(online_label[0:max_length])
            final_feats.append(final_feat[0:max_length])
            rider_feats.append(rider_feat[0:max_length])
            distance_labels.append(distance_label[0:max_length])
            lat_lng_labels.append(lat_lng_label[0:max_length])
            order_ids.append(order_id[0:max_length])
        return [labels,online_preds,rank_labels, online_labels,final_feats,rider_feats,distance_labels,lat_lng_labels,order_ids]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

if __name__ == "__main__":
    data_path = "./sample_data.csv"
    train_raw = load_raw_data(data_path,0,0)
    word2id, id2word = pickle.load(open('/home/longzhangchao/data//map.txt', "r"))
    train_data = prepare_dataset(train_raw)
    train_manager = BatchManager(train_data, 32,12)

