## tensorflow读取机制图解

我们必须要把数据先读入后才能进行计算，假设读入用时0.1s，计算用时0.9s，那么就意味着每过1s，GPU都会有0.1s无事可做，这就大大降低了运算的效率。

解决这个问题方法就是将读入数据和计算分别放在两个线程中，将数据读入内存的一个队列，如下图所示：

![image](http://upload-images.jianshu.io/upload_images/2240881-7922ded69867852b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

读取线程源源不断地将文件系统中的图片读入到一个内存的队列中，而负责计算的是另一个线程，计算需要数据时，直接从内存队列中取就可以了。这样就可以解决GPU因为IO而空闲的问题！

在tensorflow中，为了方便管理，在内存队列前又添加了一层所谓的“文件名队列”。tensorflow使用文件名队列+内存队列双队列的形式读入文件，可以很好地管理epoch。下面我们用图片的形式来说明这个机制的运行方式。

![image](http://upload-images.jianshu.io/upload_images/2240881-8da2152029948191.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/2240881-ddc2df2978d8cea4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如果再尝试读入，系统由于检测到了“结束”，就会自动抛出一个异常（OutOfRange）。外部捕捉到这个异常后就可以结束程序了。这就是tensorflow中读取数据的基本机制。

## tensorflow读取数据机制的对应函数

对于文件名队列，我们使用tf.train.string_input_producer函数。这个函数需要传入一个文件名list，系统会自动将它转为一个文件名队列。tf.train.string_input_producer还有两个重要的参数，一个是num_epochs，表示epoch数。另外一个就是shuffle是指在一个epoch内文件的顺序是否被打乱。

在tensorflow中，内存队列不需要我们自己建立，我们只需要使用reader对象从文件名队列中读取数据就可以了。

在我们使用tf.train.string_input_producer创建文件名队列后，整个系统其实还是处于“停滞状态”的，也就是说，我们文件名并没有真正被加入到队列中，此时如果我们开始计算，因为内存队列中什么也没有，计算单元就会一直等待，导致整个系统被阻塞。使用tf.train.start_queue_runners之后，才会启动填充队列的线程，这时系统就不再“停滞”。此后计算单元就可以拿到数据并进行计算，整个程序也就跑起来了。

reader每次读取一张图片并保存。
```python
import tensorflow as tf 

# 新建一个Session
with tf.Session() as sess:
    # 我们要读三幅图片A.jpg, B.jpg, C.jpg
    filename = ['A.jpg', 'B.jpg', 'C.jpg']
    # string_input_producer会产生一个文件名队列
    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=5)
    # reader从文件名队列中读数据。对应的方法是reader.read
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
    tf.local_variables_initializer().run()
    # 使用start_queue_runners之后，才会开始填充队列
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        # 获取图片数据并保存
        image_data = sess.run(value)
        with open('read/test_%d.jpg' % i, 'wb') as f:
            f.write(image_data)
```
## demo
好了，结合之前的[tensorflow学习笔记-队列和线程管理](https://www.jianshu.com/p/7fe829553b52), 我们就可以写个像样的多线程队列读取文件的代码了。
文件准备：
```shell 
$ echo -e "Alpha1,A1\nAlpha2,A2\nAlpha3,A3" > A.csv
$ echo -e "Bee1,B1\nBee2,B2\nBee3,B3" > B.csv
$ echo -e "Sea1,C1\nSea2,C2\nSea3,C3" > C.csv
$ cat A.csv
Alpha1,A1
Alpha2,A2
Alpha3,A3
```
```python
import tensorflow as tf
# 生成一个先入先出队列和一个QueueRunner
filenames = ['A.csv', 'B.csv', 'C.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
# 定义Reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# 定义Decoder
example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
# 运行Graph
with tf.Session() as sess:
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。
    for i in range(10):
        print example.eval()   #取样本的时候，一个Reader先从文件名队列中取出文件名，读出数据，Decoder解析后进入样本队列。
    coord.request_stop()
    coord.join(threads)
# outpt
Alpha1
Alpha2
Alpha3
Bee1
Bee2
Bee3
Sea1
Sea2
Sea3
Alpha1
```