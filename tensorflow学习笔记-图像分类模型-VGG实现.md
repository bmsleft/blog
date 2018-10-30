在之前的tf学习笔记系列中已经吧AlexNet的实现做了记录，这里继续，实现一下VGG。
参考之前分享过的论文详解log：[深度学习VGG模型核心拆解](https://blog.csdn.net/qq_40027052/article/details/79015827)

不过在自己实现之前，先来看看slim里的实现是什么样子的。
```python
def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
```
怎么样，很简洁的吧。对的，由于slim封装了很多基础操作，可以精简代码，而且，slim.repeat()的API让创建多个相同layer变得更容易。

如果是我们自己封装实现的话，可以借鉴slim这种方式。
这里沿用之前AlexNet采用的方式来实现一下VGG16的基础版本。对比一下slim里最后三层都用卷积层替代了。
```python

import tensorflow as tf

class VGG16(object):
    '''
    #use like this:
    model = VGG16(input, num_classes, keep_prob, is_training)
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
        with tf.name_scope('vgg16'):
            x = conv('conv1_1', self.INPUT, 64)
            x = conv('conv1_2', x, 64)
            x = max_pool('pool1', x)

            x = conv('conv2_1', x, 128)
            x = conv('conv2_2', x, 128)
            x = max_pool('pool2', x)

            x = conv('conv3_1', x, 256)
            x = conv('conv3_2', x, 256)
            x = conv('conv3_3', x, 256)
            x = max_pool('pool3', x)

            x = conv('conv4_1', x, 512)
            x = conv('conv4_2', x, 512)
            x = conv('conv4_3', x, 512)
            x = max_pool('pool4', x)

            x = conv('conv5_1', x, 512)
            x = conv('conv5_2', x, 512)
            x = conv('conv5_3', x, 512)
            x = max_pool('pool5', x)

            x = fc('fc6', x, 4096)
            x = dropout(x, self.KEEP_PROB)

            x = fc('fc7', x, 4096)
            x = dropout(x, self.KEEP_PROB)

            self.fc8 = fc('fc8', x, self.NUM_CLASSES, is_relu=False)


def conv(name, input, num_filters, filter_height=3, filter_width=3, stride_x=1, stride_y=1, padding='SAME' ):
    '''
     先定义conv的通用模式
    '''
    input_channels = int(input.get_shape()[-1])

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])
        conv = tf.nn.conv2d(input, weights,
                            strides=[1, stride_y, stride_x, 1],
                            padding=padding)
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(name, input, num_output, is_relu=True, is_trainable=True):
    '''定义全连接层'''
    shape = input.get_shape()
    if len(shape) == 4:
        num_input = shape[1].value * shape[2].value * shape[3].value
    else:
        num_input = shape[-1].value

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_input, num_output], trainable=is_trainable)
        biases = tf.get_variable('biases', [num_output], trainable=is_trainable)

        flat_x = tf.reshape(input, [-1, num_input])
        act = tf.nn.xw_plus_b(flat_x, weights, biases, name=scope.name)
        if is_relu:
            return tf.nn.relu(act, name=scope.name)
        else:
            return act


def max_pool(name, input, filter_height=2, filter_width=2, stride_x=2, stride_y=2, padding='SAME'):
    return tf.nn.max_pool(input,
                          ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_x, stride_y, 1],
                          padding=padding,
                          name=name)


def dropout(input, keep_prob=0.5, is_training=True):
    if is_training:
        return tf.nn.dropout(input, keep_prob)
    else:
        return input
```