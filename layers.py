# -*- coding:utf-8 -*-
import six
import tensorflow as tf
from tensorflow.contrib import layers


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def ff(inputs, num_units, l2_scale=0.0,scope="positionwise_feedforward"):
    '''position-wise feed forward net

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    regularizer = layers.l2_regularizer(l2_scale)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE,regularizer=regularizer):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = layer_norm(outputs)

    return outputs

def mask(inputs, queries=None, keys=None,mask_=None,num_heads=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (N, T_q, T_k)
    queries: 3d tensor. (N, T_q, d)
    keys: 3d tensor. (N, T_k, d)
    e.g.,
    >> queries = tf.constant([[[1.],
                        [2.],
                        [0.]]], tf.float32) # (1, 3, 1)
    >> keys = tf.constant([[[4.],
                     [0.]]], tf.float32)  # (1, 2, 1)
    >> inputs = tf.constant([[[4., 0.],
                               [8., 0.],
                               [0., 0.]]], tf.float32)
    >> mask(inputs, queries, keys, "key")
    array([[[ 4.0000000e+00, -4.2949673e+09],
        [ 8.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
    >> inputs = tf.constant([[[1., 0.],
                             [1., 0.],
                              [1., 0.]]], tf.float32)
    >> mask(inputs, queries, keys, "query")
    array([[[1., 0.],
        [1., 0.],
        [0., 0.]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        if mask_ is None:
            # Generate masks
            masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
            masks = tf.expand_dims(masks, 1) # (N, 1, T_k)
            masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)
            masks=masks * tf.transpose(masks, [0, 2, 1])
        else:
            masks=tf.tile(mask_,[num_heads,1,1])
        # Apply masks to inputs
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
    elif type in ("q", "query", "queries"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

        # Apply masks to inputs
        outputs = inputs*masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


def scaled_dot_product_attention(Q, K, V,mask_,num_heads,
                                 causality=False, dropout_rate=0.,train=False,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_k].
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs_raw = mask(outputs, Q, K,mask_,num_heads,type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax

        outputs = tf.nn.softmax(outputs_raw)
        attention = tf.nn.softmax(outputs_raw)
        # attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # # query masking
        # outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate,training=train)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs,attention,outputs_raw


def multihead_attention(queries, keys, values,
                        mask_,
                        num_heads=8,
                        dropout_rate=0,
                        l2_scale=0.0,
                        train=False,
                        causality=False,
                        scope="multihead_attention"):
    '''multihead attention
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    train:isTrain
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    d_model = queries.get_shape().as_list()[-1]
    regularizer = layers.l2_regularizer(l2_scale)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE,regularizer=regularizer):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs,attention,attention_raw = scaled_dot_product_attention(Q_, K_, V_,mask_,num_heads, causality, dropout_rate,train)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)
        attention = tf.reduce_mean(tf.split(attention, num_heads, axis=0), axis=0)  # (N, T_q, d_model)
        attention_raw = tf.reduce_mean(tf.split(attention_raw, num_heads, axis=0), axis=0)  # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = layer_norm(outputs)

    return outputs,attention,attention_raw
def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def dense_connect(name, input, out_dim, keep_prob=None, l2_scale=0.0):
  """
  全连接层基础组件
  :param name:
  :param input: inpout tensor
  :param out_dim:
  :param keep_prob:
  :return: output tensor
  """

  regularizer =layers.l2_regularizer(l2_scale)

  with tf.variable_scope("dense_connect",regularizer=regularizer):
      batch_size,len,dim=input.shape.as_list()
      W_fc1 = tf.Variable(
          tf.truncated_normal([dim, out_dim], stddev=0.1), name='w_' + name)
      input=tf.reshape(input,shape=[-1,dim])
      b_fc1 = tf.Variable(tf.constant(0., shape=[1]), name='b_' + name)
      output = tf.matmul(input, W_fc1) + b_fc1
      if keep_prob is not None:
          output = tf.nn.dropout(output, keep_prob)
      output=tf.reshape(output,[-1,len,out_dim])
      return output

def fm_layer(feat_index, feat_value, fm_keep_prob, config):
  """FM模块实现"""
  embeddings = tf.Variable(tf.random_normal([config.num_feat, config.embedding_size], 0.0, 0.01),
                           name="feature_embeddings")  # # feature_size * K
  input_embeddings = tf.nn.embedding_lookup(embeddings,
                                            feat_index)  # Vm*Xm, <Vm,Vn>XmXn = <VmXm,VnXn>  #None * F * K
  feat_value = tf.reshape(feat_value, shape=[-1, config.num_feat, 1])

  # 一阶部分,<W,X>
  first_order_weight = tf.Variable(tf.random_uniform([config.num_feat, 1], 0.0, 1.0),
                                   name="feature_bias")  # feature_size * 1
  first_order_weight = tf.nn.embedding_lookup(first_order_weight, feat_index)  # None * F * 1
  first_order_output = tf.reduce_sum(tf.multiply(first_order_weight, feat_value), 2)  # None*F
  first_order_output = tf.nn.dropout(first_order_output, fm_keep_prob)

  # 二阶特征组合,<Vm,Vn>XmXn=0.5*((VmXm+VmXm)^2-((VmXm)^2+(VnXn)^2))
  # 计算和的平方
  share_input_embeddings = tf.multiply(input_embeddings, feat_value)  # vk*xj    None*F*K

  summed_vx = tf.reduce_sum(share_input_embeddings, 1)  # None*K
  summed_square_vx = tf.square(summed_vx)
  # 计算平方和
  squared_vx = tf.square(share_input_embeddings)
  squared_sum_vx = tf.reduce_sum(squared_vx, 1)  # None*K
  # 两项相减
  second_order_output = 0.5 * tf.subtract(summed_square_vx, squared_sum_vx)
  second_order_output = tf.nn.dropout(second_order_output, fm_keep_prob)
  concat_input = tf.concat([first_order_output, second_order_output], axis=1)

  return concat_input


def customized_loss(labels, predictions, threshold_1=-0.3, threshold_2=0.3, penalty_1=1.0, penalty_2=1.0):
    """
    根据业务定制化损失函数
    :param labels: 实际值 Tensor
    :param predictions: 预测值 Tensor
    :param threshold_1: 偏差阈值1
    :param threshold_2: 偏差阈值2
    :param penalty_1: 预测值大于实际值(threshold_1倍)时的惩罚系数
    :param penalty_2: 预测值小于实际值(threshold_2倍)时的惩罚系数
    :return: loss
    """
    p_err = labels-predictions
    condition1 = tf.less(p_err, threshold_1)
    condition2 = tf.greater_equal(p_err, threshold_2)
    condition3 = 1-tf.add(tf.cast(condition1,tf.int32), tf.cast(condition2,tf.int32))
    residual = tf.abs(predictions - labels)
    res_less = tf.where(condition1, residual*penalty_1, tf.zeros_like(residual))
    res_more = tf.where(condition2, residual*penalty_2, tf.zeros_like(residual))
    res_medium = tf.where(tf.cast(condition3,bool), residual, tf.zeros_like(residual))
    res_final = res_less + res_medium + res_more
    return res_final

