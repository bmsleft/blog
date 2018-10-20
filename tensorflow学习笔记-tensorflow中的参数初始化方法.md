1. 初始化为常量

   tf中使用tf.constant_initializer(value)类生成一个初始值为常量value的tensor对象。

   constant_initializer类的构造函数定义：

   ```python
   def __init__(self, value=0, dtype=dtypes.float32, verify_shape=False):
       self.value = value
       self.dtype = dtypes.as_dtype(dtype)
       self._verify_shape = verify_shape
   ```

   - value：指定的常量

   - dtype： 数据类型

   - verify_shape： 是否可以调整tensor的形状，默认可以调整

   ```python
   import tensorflow as tf
   
   value = [0, 1, 2, 3, 4, 5, 6, 7]
   init = tf.constant_initializer(value)
   with tf.Session() as sess:
     x = tf.get_variable('x', shape=[8], initializer=init)
     x.initializer.run()
     print(x.eval())
   #output:
   #[ 0.  1.  2.  3.  4.  5.  6.  7.]
   ```

   神经网络中经常使用常量初始化方法来初始化偏置项。



   当初始化一个维数很多的常量时，一个一个指定每个维数上的值很不方便，tf提供了 tf.zeros_initializer() 和 tf.ones_initializer() 类，分别用来初始化全0和全1的tensor对象。

   ```python
   import tensorflow as tf
   init_zeros=tf.zeros_initializer()
   init_ones = tf.ones_initializer
   with tf.Session() as sess:
     x = tf.get_variable('x', shape=[8], initializer=init_zeros)
     y = tf.get_variable('y', shape=[8], initializer=init_ones)
     x.initializer.run()
     y.initializer.run()
     print(x.eval())
     print(y.eval())
   #output:
   # [ 0.  0.  0.  0.  0.  0.  0.  0.]
   # [ 1.  1.  1.  1.  1.  1.  1.  1.]
   ```

2. 初始化为正太分布

   初始化参数为正太分布在神经网络中应用的最多，可以初始化为标准正太分布和截断正太分布。

   tf中使用 tf.random_normal_initializer() 类来生成一组符合标准正太分布的tensor。

   tf中使用 tf.truncated_normal_initializer() 类来生成一组符合截断正太分布的tensor。

   tf.random_normal_initializer 类和 tf.truncated_normal_initializer 的构造函数定义：

   ```python
   def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=dtypes.float32):
       self.mean = mean
       self.stddev = stddev
       self.seed = seed
       self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))
   ```

   - mean： 正太分布的均值，默认值0
   - stddev： 正太分布的标准差，默认值1
   - seed： 随机数种子，指定seed的值可以每次都生成同样的数据
   - dtype： 数据类型

   ```python
   import tensorflow as tf
    
   init_random = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
   init_truncated = tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
    
    
   with tf.Session() as sess:
    
     x = tf.get_variable('x', shape=[10], initializer=init_random)
     y = tf.get_variable('y', shape=[10], initializer=init_truncated)
     x.initializer.run()
     y.initializer.run()
    
     print(x.eval())
     print(y.eval())
    
    
   #output:
   # [-0.40236568 -0.35864913 -0.94253045 -0.40153521  0.1552504   1.16989613
   #   0.43091929 -0.31410623  0.70080078 -0.9620409 ]
   # [ 0.18356581 -0.06860946 -0.55245203  1.08850253 -1.13627422 -0.1006074
   #   0.65564936  0.03948414  0.86558545 -0.4964745 ]
   ```


3. 初始化为均匀分布

   tf中使用 tf.random_uniform_initializer 类来生成一组符合均匀分布的tensor。 tf.random_uniform_initializer类构造函数定义：

   ```python
   def __init__(self, minval=0, maxval=None, seed=None, dtype=dtypes.float32):
       self.minval = minval
       self.maxval = maxval
       self.seed = seed
       self.dtype = dtypes.as_dtype(dtype)
   ```

   - minval: 最小值
   - maxval： 最大值
   - seed：随机数种子
   - dtype： 数据类型

   ```python
   import tensorflow as tf
    
   init_uniform = tf.random_uniform_initializer(minval=0, maxval=10, seed=None, dtype=tf.float32)
    
   with tf.Session() as sess:
     x = tf.get_variable('x', shape=[10], initializer=init_uniform)
     x.initializer.run() 
     print(x.eval())
    
   # output:
   # [ 6.93343639  9.41196823  5.54009819  1.38017178  1.78720832  5.38881063
   #   3.39674473  8.12443542  0.62157512  8.36026382]
   ```

   从输出可以看到，均匀分布生成的随机数并不是从小到大或者从大到小均匀分布的，这里均匀分布的意义是每次从一组服从均匀分布的数里边随机抽取一个数。

   tf中另一个生成均匀分布的类是 **tf.uniform_unit_scaling_initializer()**，构造函数是

   ```python
   def __init__(self, factor=1.0, seed=None, dtype=dtypes.float32):
       self.factor = factor
       self.seed = seed
       self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))
   ```

   同样都是生成均匀分布，tf.uniform_unit_scaling_initializer 跟 tf.random_uniform_initializer 不同的地方是前者不需要指定最大最小值，是通过公式计算出来的：

   ```python
   max_val = math.sqrt(3 / input_size) * factor
   min_val = -max_val
   ```

   input_size是生成数据的维度，factor是系数。

   ```python
   import tensorflow as tf
    
   init_uniform_unit = tf.uniform_unit_scaling_initializer(factor=1.0, seed=None, dtype=tf.float32)
    
   with tf.Session() as sess: 
     x = tf.get_variable('x', shape=[10],    initializer=init_uniform_unit)
     x.initializer.run()
     print(x.eval())
    
   # output:
   # [-1.65964031  0.59797513 -0.97036457 -0.68957627  1.69274557  1.2614969
   #   1.55491126  0.12639415  0.54466736 -1.56159735]
   ```

4. 初始化为变尺度正太、均匀分布

   tf中tf.variance_scaling_initializer()类可以生成截断正太分布和均匀分布的tensor，增加了更多的控制参数。构造函数：

   ```python
   def __init__(self, scale=1.0,
                  mode="fan_in",
                  distribution="normal",
                  seed=None,
                  dtype=dtypes.float32):
       if scale <= 0.:
         raise ValueError("`scale` must be positive float.")
       if mode not in {"fan_in", "fan_out", "fan_avg"}:
         raise ValueError("Invalid `mode` argument:", mode)
       distribution = distribution.lower()
       if distribution not in {"normal", "uniform"}:
         raise ValueError("Invalid `distribution` argument:", distribution)
       self.scale = scale
       self.mode = mode
       self.distribution = distribution
       self.seed = seed
       self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))
   ```

   - scale: 缩放尺度
   -  mode： 有3个值可选，分别是 “fan_in”, “fan_out” 和 “fan_avg”，用于控制计算标准差 stddev的值
   - distribution： 2个值可选，”normal”或“uniform”，定义生成的tensor的分布是截断正太分布还是均匀分布

   distribution选‘normal’的时候，生成的是截断正太分布，标准差 stddev = sqrt(scale / n), n的取值根据mode的不同设置而不同：

   1. mode = "fan_in"， n为输入单元的结点数；         
   2. mode = "fan_out"，n为输出单元的结点数；
   3. mode = "fan_avg",n为输入和输出单元结点数的平均值;

   distribution选 ‘uniform’，生成均匀分布的随机数tensor，最大值 max_value和 最小值 min_value 的计算公式：

   ```python
   max_value = sqrt(3 * scale / n)
   min_value = -max_value
   ```

   ```python
   import tensorflow as tf
    
   init_variance_scaling_normal = tf.variance_scaling_initializer(scale=1.0,mode="fan_in",                                                    distribution="normal",seed=None,dtype=tf.float32)
   init_variance_scaling_uniform = tf.variance_scaling_initializer(scale=1.0,mode="fan_in",                                                    distribution="uniform",seed=None,dtype=tf.float32)
    
   with tf.Session() as sess:
     x = tf.get_variable('x', shape=[10], initializer=init_variance_scaling_normal)
     y = tf.get_variable('y', shape=[10], initializer=init_variance_scaling_uniform)
     x.initializer.run()
     y.initializer.run()
     print(x.eval())
     print(y.eval())
    
   # output:
   # [ 0.55602223  0.36556259  0.39404872 -0.11241052  0.42891756 -0.22287074
   #   0.15629818  0.56271428 -0.15364751 -0.03651841]
   # [ 0.22965753 -0.1339919  -0.21013224  0.112804   -0.49030468  0.21375734
   #   0.24524075 -0.48397955  0.02254289 -0.07996771]
   ```

5. 其他初始化方式

   - tf.orthogonal_initializer() 初始化为正交矩阵的随机数，形状最少需要是二维的
   - tf.glorot_uniform_initializer() 初始化为与输入输出节点数相关的均匀分布随机数
   - tf.glorot_normal_initializer（） 初始化为与输入输出节点数相关的截断正太分布随机数

   ```python
   import tensorflow as tf
    
   init_orthogonal = tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32)
   init_glorot_uniform = tf.glorot_uniform_initializer()
   init_glorot_normal = tf.glorot_normal_initializer()
    
   with tf.Session() as sess:
    
     x = tf.get_variable('x', shape=[4,4], initializer=init_orthogonal)
     y = tf.get_variable('y', shape=[10], initializer=init_glorot_uniform)
     z = tf.get_variable('z', shape=[10], initializer=init_glorot_normal)
    
     x.initializer.run()
     y.initializer.run()
     z.initializer.run()
    
     print(x.eval())
     print(y.eval())
     print(z.eval())
    
   # output:
   # [[ 0.41819954  0.38149482  0.82090431  0.07541249]
   #  [ 0.41401231  0.21400851 -0.38360971  0.79726893]
   #  [ 0.73776144 -0.62585676 -0.06246936 -0.24517137]
   #  [ 0.33077344  0.64572859 -0.41839844 -0.54641217]]
   # [-0.11182356  0.01995623 -0.0083192  -0.09200105  0.49967837  0.17083591
   #   0.37086374  0.09727859  0.51015782 -0.43838671]
   # [-0.50223351  0.18181904  0.43594137  0.3390047   0.61405027  0.02597036
    
   #   0.31719241  0.04096413  0.10962497 -0.13165198]
   ```









