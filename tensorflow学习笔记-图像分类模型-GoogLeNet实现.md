> 导言：对GoogLeNet的了解不是特别深入，至少算法动机还没熟练掌握。那这篇笔记就先总结一下GoogLeNet的原理以及V1-V4的演化。最后用slim中的demo学习一下用tf实现的GoogLeNet。

### GoogLeNet的理论基础

将Hebbian原理应用在神经网络上，如果数据集的概率分布可以被一个很大很稀疏的神经网络表达，那么构筑这个网络的最佳方法是逐层构筑网络：将上一层高度相关的节点聚类，并将聚类出来的每一个小簇连接到一起。

- 什么是Hebbian原理？ 
  神经反射活动的持续与重复会导致神经元连续稳定性持久提升，当两个神经元细胞A和B距离很近，并且A参与了对B重复、持续的兴奋，那么某些代谢会导致A将作为使B兴奋的细胞。总结一下:“一起发射的神经元会连在一起”,学习过程中的刺激会使神经元间突触强度增加。

- 这里我们先讨论一下为什么需要稀疏的神经网络是什么概念? 
  人脑神经元的连接是稀疏的，研究者认为大型神经网络的合理的连接方式应该也是稀疏的，稀疏结构是非常适合神经网络的一种结构，尤其是对非常大型、非常深的神经网络，可以减轻过拟合并降低计算量，例如CNN就是稀疏连接。

- 为什么CNN就是稀疏连接？ 
  在符合Hebbian原理的基础上，我们应该把相关性高的一簇神经元节点连接在一起。在普通的数据集中，这可能需要对神经元节点做聚类，但是在图片数据中，天然的就是临近区域的数据相关性高，因此相邻的像素点被卷积操作连接在一起（符合Hebbian原理），而卷积操作就是在做稀疏连接。

- 怎样构建满足Hebbian原理的网络？ 

  在CNN模型中，我们可能有多个卷积核，在同一空间位置但在不同通道的卷积核的输出结果相关性极高。我们可以使用1*1的卷积很自然的把这些相关性很高的、在同一空间位置但是不同通道的特征连接在一起。



### 动机和主要工作

一般来说，提升网络性能最直接的方式就是增加网络的大小:

1. 增加网络的深度 
2. 增加网络的宽度 

这样简单的解决办法有两个主要的缺点: 

1. 网络参数的增多，网络容易陷入过拟合中，这需要大量的训练数据，而在解决高粒度分类的问题上，高质量的训练数据成本太高; 
2. 简单的增加网络的大小，会让网络计算量增大，而增大计算量得不到充分的利用，从而造成计算资源的浪费

解决上面的两个缺点的思路：
将全连接的结构转换为稀疏结构(即使是内部卷积) 
如果数据集的概率分布可以可以有大型的稀疏的深度神经网络表示，则优化网络的方法可以是逐层的分析层输出的相关性，对相关的输出做聚类操作.

inception架构的主要思想是建立在找到可以逼近的卷积视觉网络内的最优局部稀疏结构，并可以通过易实现的模块实现这种结构; 
使用大的卷积核在空间上会扩散更多的区域，而对应的聚类就会变少，聚类的数目随着卷积核增大而减少，为了避免这个问题，inception架构当前只使用1*1,3*3,5*5的滤波器大小，这个决策更多的是为了方便而不是必须的。 

### 做法Q&A

- **1x1卷积的作用？**
  1. 在相同尺寸的感受野中叠加更多的卷积，能提取到更丰富的特征。
  2. 使用1x1卷积进行降维，降低了计算复杂度。

- **多个尺寸上进行卷积再聚合的解释**
  1. 在直观感觉上在多个尺度上同时进行卷积，能提取到不同尺度的特征。特征更为丰富也意味着最后分类判断时更加准确。
  2. 利用稀疏矩阵分解成密集矩阵计算的原理来加快收敛速度。
  3. Hebbin赫布原理。Hebbin原理是神经科学上的一个理论，解释了在学习的过程中脑中的神经元所发生的变化，用一句话概括就是*fire togethter, wire together*。比如狗看到肉会流口水，反复刺激后，脑中识别肉的神经元会和掌管唾液分泌的神经元会相互促进，“缠绕”在一起，以后再看到肉就会更快流出口水。用在inception结构中就是要把相关性强的特征汇聚到一起。这有点类似上面的解释2，把1x1，3x3，5x5的特征分开。因为训练收敛的最终目的就是要提取出独立的特征，所以预先把相关性强的特征汇聚，就能起到加速收敛的作用。

- **为什么要用max pooling？**

  作者认为pooling也能起到提取特征的作用，所以也加入模块中。因为貌似有很多papers提到了这个pool的作用。注意这个pooling的stride=1，pooling后没有减少数据的尺寸。



### GoogLeNet的发展

- 2014年9月的《Going deeper with convolutions》提出的Inception V1.
- 2015年2月的《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》提出的Inception V2
- 2015年12月的《Rethinking the Inception Architecture for Computer Vision》提出的Inception V3
- 2016年2月的《Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning》提出的Inception V4

**Inception V1**——构建了1x1、3x3、5x5的 conv 和3x3的 pooling 的**分支网络**，同时使用 **MLPConv** 和**全局平均池化**，扩宽卷积层网络宽度，增加了网络对尺度的适应性；

**Inception V2**——提出了 **Batch Normalization**，代替 **Dropout** 和 **LRN**，其正则化的效果让大型卷积网络的训练速度加快很多倍，同时收敛后的分类准确率也可以得到大幅提高，同时学习 **VGG** 使用两个3´3的卷积核代替5´5的卷积核，在降低参数量同时提高网络学习能力；

**Inception V3**——引入了 **Factorization**，将一个较大的二维卷积拆成两个较小的一维卷积，比如将3´3卷积拆成1´3卷积和3´1卷积，一方面节约了大量参数，加速运算并减轻了过拟合，同时增加了一层非线性扩展模型表达能力，除了在 **Inception Module** 中使用分支，还在分支中使用了分支（**Network In Network In Network**）；

**Inception V4**——研究了 **Inception Module** 结合 **Residual Connection**，结合 **ResNet** 可以极大地加速训练，同时极大提升性能，在构建 Inception-ResNet 网络同时，还设计了一个更深更优化的 Inception v4 模型，能达到相媲美的性能。





### GoogLeNet的tensorflow实现

这里使用slim里的实现进行说明吧。（原谅我偷懒了，没有自己写。。。）

```python
# 生成V3网络的卷积部分
def inception_v3_base(inputs, scope=None):
  '''
  Args:
  inputs：输入的tensor
  scope：包含了函数默认参数的环境
  '''
  end_points = {} # 定义一个字典表保存某些关键节点供之后使用

  with tf.variable_scope(scope, 'InceptionV3', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], # 对三个参数设置默认值
                        stride=1, padding='VALID'):

      #  因为使用了slim以及slim.arg_scope，我们一行代码就可以定义好一个卷积层
      #  相比AlexNet使用好几行代码定义一个卷积层，或是VGGNet中专门写一个函数定义卷积层，都更加方便
      #
      # 正式定义Inception V3的网络结构。首先是前面的非Inception Module的卷积层
      # slim.conv2d函数第一个参数为输入的tensor，第二个是输出的通道数，卷积核尺寸，步长stride，padding模式

      #一共有5个卷积层，2个池化层，实现了对图片数据的尺寸压缩，并对图片特征进行了抽象
      # 299 x 299 x 3
      net = slim.conv2d(inputs, 32, [3, 3],
                        stride=2, scope='Conv2d_1a_3x3')    # 149 x 149 x 32

      net = slim.conv2d(net, 32, [3, 3],
                        scope='Conv2d_2a_3x3')      # 147 x 147 x 32

      net = slim.conv2d(net, 64, [3, 3], padding='SAME',
                        scope='Conv2d_2b_3x3')  # 147 x 147 x 64

      net = slim.max_pool2d(net, [3, 3], stride=2,
                            scope='MaxPool_3a_3x3')   # 73 x 73 x 64

      net = slim.conv2d(net, 80, [1, 1],
                        scope='Conv2d_3b_1x1')  # 73 x 73 x 80

      net = slim.conv2d(net, 192, [3, 3],
                        scope='Conv2d_4a_3x3')  # 71 x 71 x 192

      net = slim.max_pool2d(net, [3, 3], stride=2,
                            scope='MaxPool_5a_3x3') # 35 x 35 x 192


    '''
    三个连续的Inception模块组，三个Inception模块组中各自分别有多个Inception Module，这部分是Inception Module V3
    的精华所在。每个Inception模块组内部的几个Inception Mdoule结构非常相似，但是存在一些细节的不同
    '''
    # Inception blocks
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], # 设置所有模块组的默认参数
                        stride=1, padding='SAME'): # 将所有卷积层、最大池化、平均池化层步长都设置为1
      # 第一个模块组包含了三个结构类似的Inception Module
      '''    
--------------------------------------------------------    
      第一个Inception组   一共三个Inception模块
      '''
      with tf.variable_scope('Mixed_5b'): # 第一个Inception Module名称。Inception Module有四个分支
        # 第一个分支64通道的1*1卷积
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1') # 35x35x64

        # 第二个分支48通道1*1卷积，链接一个64通道的5*5卷积
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1') # 35x35x48
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5') #35x35x64

        # 第三个分支64通道1*1卷积,96的3*3,再接一个3*3
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')#35x35x96

        # 第四个分支64通道3*3平均池化,32的1*1
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1') #35*35*32
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3) # 将四个分支的输出合并在一起（第三个维度合并，即输出通道上合并）
      # 64+64+96+32 = 256
      # mixed_1: 35 x 35 x 256.
      '''
      因为这里所有层步长均为1，并且padding模式为SAME，所以图片尺寸不会缩小，但是通道数增加了。四个分支通道数之和
      64+64+96+32=256，最终输出的tensor的图片尺寸为35*35*256
      '''

      with tf.variable_scope('Mixed_5c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      # 64+64+96+64 = 288
      # mixed_2: 35 x 35 x 288.

      with tf.variable_scope('Mixed_5d'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # 64+64+96+64 = 288
      # mixed_1: 35 x 35 x 288

        '''    
        第一个Inception组结束  一共三个Inception模块 输出为:35*35*288
----------------------------------------------------------------------    
        第二个Inception组   共5个Inception模块
        '''

      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 384, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1') #17*17*384
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1') #35*35*64
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')#35*35*96
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1') #17*17*96
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3') #17*17*288
        net = tf.concat([branch_0, branch_1, branch_2], 3) # 输出尺寸定格在17 x 17 x 768
      # 384+96+288 = 768
      # mixed_3: 17 x 17 x 768.

      with tf.variable_scope('Mixed_6b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7') # 串联1*7卷积和7*1卷积合成7*7卷积，减少了参数，减轻了过拟合
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1') # 反复将7*7卷积拆分
          branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      # 192+192+192+192 = 768
      # mixed4: 17 x 17 x 768.


      with tf.variable_scope('Mixed_6c'):
        with tf.variable_scope('Branch_0'):
          '''
          我们的网络每经过一个inception module，即使输出尺寸不变，但是特征都相当于被重新精炼了一遍，
          其中丰富的卷积和非线性化对提升网络性能帮助很大。
          '''
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      # 192+192+192+192 = 768
      # mixed_5: 17 x 17 x 768.


      with tf.variable_scope('Mixed_6d'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      # 92+192+192+192 = 768
      # mixed_6: 17 x 17 x 768.

      with tf.variable_scope('Mixed_6e'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      # 92+192+192+192 = 768
      # mixed_7: 17 x 17 x 768.


        '''    
        第二个Inception组结束  一共五个Inception模块 输出为:17*17*768
----------------------------------------------------------------------    
        第三个Inception组   共3个Inception模块(带分支)
        '''
      # 将Mixed_6e存储于end_points中，作为Auxiliary Classifier辅助模型的分类
      end_points['Mixed_6e'] = net

      # 第三个inception模块组包含了三个inception module

      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')# 17*17*192
          branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3') # 8*8*320
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1') #17*17*192
          branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3') #8*8*192
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')  #8*8*768
        net = tf.concat([branch_0, branch_1, branch_2], 3) # 输出图片尺寸被缩小，通道数增加，tensor的总size在持续下降中
      # 320+192+768 = 1280
      # mixed_8: 8 x 8 x 1280.


      with tf.variable_scope('Mixed_7b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat([
              slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat([
              slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3) # 输出通道数增加到2048
      # 320+(384+384)+(384+384)+192 = 2048
      # mixed_9: 8 x 8 x 2048.



      with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat([
              slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat([
              slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
      # 320+(384+384)+(384+384)+192 = 2048
      # mixed_10: 8 x 8 x 2048.

      return net, end_points
      #Inception V3网络的核心部分，即卷积层部分就完成了
      '''
      设计inception net的重要原则是图片尺寸不断缩小，inception模块组的目的都是将空间结构简化，同时将空间信息转化为
      高阶抽象的特征信息，即将空间维度转为通道的维度。降低了计算量。Inception Module是通过组合比较简单的特征
      抽象（分支1）、比较比较复杂的特征抽象（分支2和分支3）和一个简化结构的池化层（分支4），一共四种不同程度的
      特征抽象和变换来有选择地保留不同层次的高阶特征，这样最大程度地丰富网络的表达能力。
      '''



# V3最后部分
# 全局平均池化、Softmax和Auxiliary Logits
def inception_v3(inputs,
                 num_classes=1000, # 最后需要分类的数量（比赛数据集的种类数）
                 is_training=True, # 标志是否为训练过程，只有在训练时Batch normalization和Dropout才会启用
                 dropout_keep_prob=0.8, # 节点保留比率
                 prediction_fn=slim.softmax, # 最后用来分类的函数
                 spatial_squeeze=True, # 参数标志是否对输出进行squeeze操作（去除维度数为1的维度，比如5*3*1转为5*3）
                 reuse=None, # 是否对网络和Variable进行重复使用
                 scope='InceptionV3'): # 包含函数默认参数的环境

  with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], # 定义参数默认值
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout], # 定义标志默认值
                        is_training=is_training):
      # 拿到最后一层的输出net和重要节点的字典表end_points
      net, end_points = inception_v3_base(inputs, scope=scope) # 用定义好的函数构筑整个网络的卷积部分

      # Auxiliary logits作为辅助分类的节点，对分类结果预测有很大帮助
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'): # 将卷积、最大池化、平均池化步长设置为1
        aux_logits = end_points['Mixed_6e'] # 通过end_points取到Mixed_6e
        # end_points['Mixed_6e']  --> 17x17x768
        with tf.variable_scope('AuxLogits'):
          aux_logits = slim.avg_pool2d(
                    aux_logits, [5, 5], stride=3, padding='VALID',
                    scope='AvgPool_1a_5x5') #5x5x768

          aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                   scope='Conv2d_1b_1x1') #5x5x128

          # Shape of feature map before the final layer.
          aux_logits = slim.conv2d(
              aux_logits, 768, [5, 5],
              weights_initializer=trunc_normal(0.01),
              padding='VALID', scope='Conv2d_2a_5x5')  #1x1x768

          aux_logits = slim.conv2d(
              aux_logits, num_classes, [1, 1], activation_fn=None,
              normalizer_fn=None, weights_initializer=trunc_normal(0.001),
              scope='Conv2d_2b_1x1')  # 1*1*1000

          if spatial_squeeze: # tf.squeeze消除tensor中前两个为1的维度。
            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
          end_points['AuxLogits'] = aux_logits # 最后将辅助分类节点的输出aux_logits储存到字典表end_points中

      # 处理正常的分类预测逻辑
      # Final pooling and prediction
      with tf.variable_scope('Logits'):
        # net --> 8x8x2048
        net = slim.avg_pool2d(net, [8, 8], padding='VALID',
                              scope='AvgPool_1a_8x8') #1x1x2048

        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net

        # 激活函数和规范化函数设为空
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1') # 1x1x1000
        if spatial_squeeze: # tf.squeeze去除输出tensor中维度为1的节点
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions') # Softmax对结果进行分类预测
  return logits, end_points # 最后返回logits和包含辅助节点的end_points

```





Refs:
 [TensorFlow实战：Chapter-5（CNN-3-经典卷积神经网络（GoogleNet）)](https://blog.csdn.net/u011974639/article/details/76460849?utm_source=blogxgwz1)
[深入理解GoogLeNet结构（原创）](https://zhuanlan.zhihu.com/p/32702031)
[GoogLeNet 之 Inception(V1-V4)](https://blog.csdn.net/hejin_some/article/details/78636586)

