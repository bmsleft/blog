这篇笔记主要记录一下学习tensorflow cifar-10图像分类的示例代码。

[TOC]

### 数据介绍

**Cifar-10**是由 Hinton 的两个大弟子 Alex Krizhevsky、Ilya Sutskever 收集的一个用于普适物体识别的数据集。Cifar 是加拿大政府牵头投资的一个先进科学项目研究所。Hinton、Bengio和他的学生在2004年拿到了 Cifar 投资的少量资金，建立了神经计算和自适应感知项目。这个项目结集了不少计算机科学家、生物学家、电气工程师、神经科学家、物理学家、心理学家，加速推动了 Deep Learning  的进程。从这个阵容来看，DL 已经和 ML 系的数据挖掘分的很远了。Deep Learning 强调的是自适应感知和人工智能，是计算机与神经科学交叉；Data Mining 强调的是高速、大数据、统计数学分析，是计算机和数学的交叉。

cifar-10分类数据集为60000张32 * 32的彩色图片，总共有10个类别，其中50000张训练集，10000张测试集，官网地址[http://www.cs.toronto.edu/~kriz/cifar.html](http://www.cs.toronto.edu/~kriz/cifar.html)，提供数据集下载 数据集中图片诸如以下
![20171108170500783.png](https://upload-images.jianshu.io/upload_images/2240881-364253bd94d6371a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
这里重点说明数据集的存储格式。下载解压后为5个batch（图中data_batch_1,2,3,4,5）的训练集，1个batch（图中test_batch）的测试集。每个batch中10000张图片。
![20171108171349118.png](https://upload-images.jianshu.io/upload_images/2240881-464b907a36e21581.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
每个文件中数据存储格式为dict字典，键值为b’data’的为图片数据，是一个10000 * 3072（32 * 32 * 3）的numpy向量，10000表示图片张数，3072中前1024个表示Red通道数据，中间1024个表示Green通道数据，最后1024个表示Blue通道数据，数据范围是0-255，表示像素点灰度。键值为b’labels’表示对应的标签，是一个长度为10000的list，数据范围是0-9，分别表示10个类别。 
另外要说明的是卷积滤波器卷积的是32 * 32 * 3格式的数据，32 * 32代表图片一个通道格式，3表示RGB 3个通道，然而依据其数据表示格式，在 reshape 3072维度的向量的时候必须首先reshape成3 * 32 * 32格式的向量，否则会破坏图片原本格式，怎么办呢，转置！类似于矩阵的转置，三维向量也有转置，tensorflow提供transpose方法对三维向量作转置。

### 网络结构
先介绍一下示例代码中的文件构成：
本文将使用的代码结果和网络结构：
| 文件                       | 说明                                   |
| -------------------------- | -------------------------------------- |
| cifar10_input.py           | 读取本地CIFAR-10的二进制文件格式的内容 |
| cifar10.py                 | 建立CIFAR-10的模型                     |
| cifar10_train.py           | 在CPU或GPU上训练CIFAR-10的模型         |
| cifar10_multi_gpu_train.py | 在多GPU上训练CIFAR-10的模型。          |
| cifar10_eval.py            | 评估CIFAR-10模型的预测性能             |
示例中的网络结构比较简单，
![20170322111837119.png](https://upload-images.jianshu.io/upload_images/2240881-6b66cb6f3528ce09.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 模型输入

输入模型是通过 cifar10_input.inputs() 和 cifar10_input.distorted_inputs() 函数建立起来的，这2个函数会从 CIFAR-10 二进制文件中读取图片文件，具体实现定义在 cifar10_input.py 中，使用的数据为 [CIFAR-10 page](http://www.cs.toronto.edu/~kriz/cifar.html) 下的162M 的二进制文件，由于每个图片的存储字节数是固定的，因此可以使用 tf.FixedLengthRecordReader 函数。

载入图像数据后，通过以下流程进行数据增广：
1.  统一裁剪到24x24像素大小，裁剪中央区域用于评估或随机裁剪用于训练；
2.  对图像进行随机的左右翻转；
3.  随机变换图像的亮度；
4.  随机变换图像的对比度；
5.  图片会进行近似的白化处理。
    其中，白化(whitening)处理或者叫标准化(standardization)处理，是对图片数据减去均值，除以方差，保证数据零均值，方差为1，如此降低输入图像的冗余性，尽量去除输入特征间的相关性，使得网络对图片的动态范围变化不敏感。
```python
    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)
```
从磁盘上加载图像并进行变换需要花费不少的处理时间。为了避免这些操作减慢训练过程，使用16个独立的线程中并行进行这些操作，这16个线程被连续的安排在一个 TensorFlow 队列中，最后返回预处理后封装好的tensor，每次执行都会生成一个 batch_size 数量的样本 [images，labels]。测试数据使用cifar10_input.inputs() 函数生成，测试数据不需要对图片进行翻转或修改亮度、对比度，需要裁剪图片正中间的24*24大小的区块，并进行数据标准化操作。
```python
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  ...... 
  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  return images, tf.reshape(label_batch, [batch_size])
```
#### 模型构建
在建立模型之前，我们构造 weight 的构造函数 _variable_with_weight_decay(name, shape, stddev, wd)，其中 wd 用于向 losses 添加L2正则化，可以防止过拟合，提高泛化能力：
```python
def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var
```
然后我们开始建立网络，第一层卷积层的 weight 不进行 L2正则，因此 kernel(wd) 这一项设为0，建立值为0的 biases，conv1的结果由 ReLu 激活，由 _activation_summary() 进行汇总；然后建立第一层池化层，最大池化尺寸和步长不一致可以增加数据的丰富性；最后建立 LRN 层，LRN层模仿了生物神经系统的"侧抑制"机制，对局部神经元的活动创建竞争环境，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力，LRN 对 Relu 这种没有上限边界的激活函数会比较有用，因为它会从附近的多个卷积核的响应中挑选比较大的反馈，但不适合 sigmoid 这种有固定边界并且能抑制过大的激活函数。
```python
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
```
第二层卷积层与第一层，除了输入参数的改变之外，将 biases 值全部初始化为0.1，调换最大池化和 LRN 层的顺序，先进行LRN，再使用最大池化层。
```python

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
```

第三层全连接层 ，需要先把前面的卷积层的输出结果全部 flatten，使用 tf.reshape 函数将每个样本都变为一维向量，使用 get_shape 函数获取数据扁平化之后的长度；然后对全连接层的 weights 和 biases 进行初始化，为了防止全连接层过拟合，设置一个非零的 wd 值0.004，让这一层的所有参数都被 L2正则所约束，最后依然使用 Relu 激活函数进行非线性化。同理，可以建立第四层全连接层。
```python
  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)
```
最后的 softmax_linear 层，先创建这一层的 weights 和 biases，不添加L2正则化。在这个模型中，不像之前的例子使用 sotfmax 输出最后的结果，因为将 softmax 的操作放在来计算 loss 的部分，将 softmax_linear 的线性返回值 logits 与 labels 计算 loss
```python
  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=None)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)
```
然后计算loss：先计算交叉熵，然后在加上L2 loss
```python
def loss(logits, labels):
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')
```

#### 模型训练
在定义 loss 之后，我们需要定义接受 loss 并返回 train op 的 train()。
首先定义学习率（learning rate），并设置随迭代次数衰减，并进行 summary：
```python
  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)
```
此外，我们对 loss 生成滑动均值和汇总，通过使用指数衰减，来维护变量的滑动均值(Moving Average)。当训练模型时，维护训练参数的滑动均值是有好处的，在测试过程中使用滑动参数比最终训练的参数值本身，会提高模型的实际性能即准确率。apply() 方法会添加 trained variables 的 shadow copies，并添加操作来维护变量的滑动均值到 shadow copies。 average() 方法可以访问 shadow variables，在创建 evaluation model 时非常有用。滑动均值是通过指数衰减计算得到的，shadow variable 的初始化值和 trained variables 相同，其更新公式为 shadow_variable = decay * shadow_variable + (1 - decay) * variable。(关于ema，可参考之前的这篇[tensorflow学习笔记-ExponentialMovingAverage](https://www.jianshu.com/p/05dd5cdacf32))
```python
def _add_loss_summaries(total_loss):
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op
```
然后，我们定义训练方法与目标，tf.control_dependencies 是一个 context manager，控制节点执行顺序，先执行[ ]中的操作，再执行 context 中的操作。计算并应用梯度，最后，动态调整衰减率，返回模型参数变量的滑动更新操作即 train op
```python
 # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

# Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op
```
#### 训练过程
```python
def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,  #这里会自动保存checkpoint
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
```


