------------
### ExponentialMovingAverage

Some training algorithms, such as GradientDescent and Momentum often benefit from maintaining a moving average of variables during optimization. Using the moving averages for evaluations often improve results significantly. 
tensorflow 官网上对于这个方法功能的介绍。GradientDescent 和 Momentum 方式的训练 都能够从 ExponentialMovingAverage 方法中获益。

什么是MovingAverage?  我的理解就是一段时间窗口内的这个变量的历史平均值。

### tensorflow 中的 ExponentialMovingAverage
官方文档中的公式: 
shadowVariable=decay∗shadowVariable+(1−decay)∗variable
官网的example：
```python
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...
...
# Create an op that applies the optimizer.  This is what we usually
# would use as a training op.
opt_op = opt.minimize(my_loss, [var0, var1])

# Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

# Create the shadow variables, and add ops to maintain moving averages
# of var0 and var1.
maintain_averages_op = ema.apply([var0, var1])

# Create an op that will update the moving averages after each training
# step.  This is what we will use in place of the usual training op.
with tf.control_dependencies([opt_op]):
    training_op = tf.group(maintain_averages_op)
    # run这个op获取当前时刻 ema_value
    get_var0_average_op = ema.average(var0)
```
### 使用 ExponentialMovingAveraged parameters
假设我们使用了ExponentialMovingAverage方法训练了神经网络， 在test阶段，如何使用 ExponentialMovingAveraged parameters呢？ 官网也给出了答案 
```python
# Create a Saver that loads variables from their saved shadow values.
shadow_var0_name = ema.average_name(var0)
shadow_var1_name = ema.average_name(var1)
saver = tf.train.Saver({shadow_var0_name: var0, shadow_var1_name: var1})
saver.restore(...checkpoint filename...)
# var0 and var1 now hold the moving average values
```
或者
```python
#Returns a map of names to Variables to restore.
variables_to_restore = ema.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
...
saver.restore(...checkpoint filename...)
```

转载： [CSDN Blog](https://blog.csdn.net/u012436149/article/details/56484572 "CSDN Blog")

---------