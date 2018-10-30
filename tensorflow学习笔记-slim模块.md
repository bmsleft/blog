> slim是TF推出的一个轻量级的high-level API of TensorFlow，用起来很方便，啊哈哈～


TF-Slim 是 tensorflow 中定义、训练和评估复杂模型的轻量级库。tf-slim中的组件可以轻易地和原生 tensorflow 框架以及例如 tf.contrib.learn 这样的框架进行整合。

Github：[https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)
可以看到，slim代码是放在contrib下的，还没转正。不过可以先玩起来哈～

在TF的modles仓库 [https://github.com/tensorflow/models/tree/master/research/slim](https://github.com/tensorflow/models/tree/master/research/slim)下，我们可以看到slim的具体应用。值得高兴的事，slim中已经实现了很多经典的图像分类模型，并提供各个模型在ImageNet上的预训练结果，我们可以直接下载微调参数就可以应用到自己的数据上了，非常nice～ 
不过这个目录下的代码不是official的，不一定能适用最新版的TF，可能需要自己修改一下代码。

这里转载一下slim的api详细的介绍：
[tensorflow中slim模块api介绍](https://blog.csdn.net/guvcolie/article/details/77686555)



