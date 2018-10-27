> 之前一篇文章中总结了CNN中图像分类的经典模型，包括论文解读和分析，但是不写个代码搞一把总觉得虚～ 啊哈哈 这个系列里准备把这些个经典模型用tensorflow实现一下。

参考之前引用的blog：[深度学习AlexNet模型详细分析](https://blog.csdn.net/zyqdragon/article/details/72353420)

上代码吧。参照着模型看更好读一些。
```python
'''
图像分类模型的tensorflow实现之--AlexNet

Tensorflow Version: 1.4
Python Version: 3.6

Refs: https://blog.csdn.net/zyqdragon/article/details/72353420 
bms
2018-10-25
'''


import tensorflow as tf
import numpy as np

class AlexNet(object):
    '''
    #use like this:
    model = AlexNet(input, num_classes, keep_prob, is_training)
    score = model.fc8
    # then you can get loss op using score
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))
    '''
    def __init__(self, input, num_classes, keep_prob=0.5, is_training=True):
        self.INPUT = input
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.IS_TRAINING = is_training
        self.default_image_size = 224

        self.create()

    def create(self):
        # 1st Layer : conv -> pool -> lrn
        conv1 = conv(self.INPUT, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

        # 2nd Layer: Conv -> Pool -> Lrn
        conv2 = conv(norm1, 5, 5, 256, 1, 1, name='conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

        # 3rd Layer: Conv
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv
        conv4 = conv(conv3, 3, 3, 384, 1, 1, name='conv4')

        # 5th Layer: Conv  -> Pool
        conv5 = conv(conv4, 3, 3, 256, 1, 1, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB, is_training=self.IS_TRAINING)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB, is_training=self.IS_TRAINING)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, name='fc8', is_relu=False)



def conv(input, filter_height, filter_width, num_filters, stride_x, stride_y, name, padding='SAME' ):
    '''
     先定义conv的通用模式
    '''
    input_channels = int(input.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])
        conv = tf.nn.conv2d(input, weights,
                            strides=[1, stride_y, stride_x, 1],
                            padding=padding)
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(input, num_input, num_output, name, is_relu=True):
    '''定义全连接层'''
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_input, num_output], trainable=True)
        biases = tf.get_variable('biases', [num_output], trainable=True)

        act = tf.nn.xw_plus_b(input, weights, biases, name=scope.name)
        if is_relu:
            return tf.nn.relu(act, name=scope.name)
        else:
            return act


def max_pool(input, filter_height, filter_width, stride_x, stride_y, name, padding='SAME'):
    return tf.nn.max_pool(input,
                          ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_x, stride_y, 1],
                          padding=padding,
                          name=name)


def lrn(input, radius=2, alpha=2e-05, beta=0.75, bias=1.0, name=''):
    return tf.nn.local_response_normalization(input,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)


def dropout(input, keep_prob=0.5, is_training=True):
    if is_training:
        return tf.nn.dropout(input, keep_prob)
    else:
        return input
```

嗯 实现起来有点麻烦啊。看一下TF的slim的实现(https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py)：
```python
def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2',
               global_pool=False):
  """AlexNet version 2.
  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224 or set
        global_pool=True. To use in fully convolutional mode, set
        spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: the number of predicted classes. If 0 or None, the logits layer
    is omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      logits. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original AlexNet.)
  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0
      or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
      net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
      net = slim.conv2d(net, 192, [5, 5], scope='conv2')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = slim.conv2d(net, 384, [3, 3], scope='conv3')
      net = slim.conv2d(net, 384, [3, 3], scope='conv4')
      net = slim.conv2d(net, 256, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

      # Use conv2d instead of fully_connected layers.
      with slim.arg_scope([slim.conv2d],
                          weights_initializer=trunc_normal(0.005),
                          biases_initializer=tf.constant_initializer(0.1)):
        net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                          scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)
        if global_pool:
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net
        if num_classes:
          net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                             scope='dropout7')
          net = slim.conv2d(net, num_classes, [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            biases_initializer=tf.zeros_initializer(),
                            scope='fc8')
          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
          end_points[sc.name + '/fc8'] = net
      return net, end_points
alexnet_v2.default_image_size = 224
```

只看模型构建部分，如此简洁。好吧，slim还是很方便的。
注意这个AlexNet是v2版本，最后三层的全连接换成了卷积层。这样输出的维度是一致的，不过由于使用了卷积，融合了多通道的信息，效果会更好些。