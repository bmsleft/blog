### 队列类型
```python
FIFOQueue()          #先入先出队列
RandomShuffleQueue() #随机队列

queue1 = tf.RandomShuffleQueue(...)
queue2 = tf.FIFOQueue(...)

##出入队列
# enqueue()、enqueue_many()
# dequeueu()、dequeue_many()

enqueue_op = queue.enqueue(example)
inputs = queue.dequeue_many(batch_size)
```

### 线程管理器Coordinator 
- should_stop()：如果线程应该停止则返回True
- request_stop(<exception>)：请求该线程停止
- join(<list of threads>)：等待被指定的子线程终止(才开始继续主线程)
- 步骤
  - 首先创建一个Coordinator对象，然后建立一些使用Coordinator对象的线程
  - 这些线程通常一直循环运行，一直到should_stop()返回True时停止
  - 任何线程都可以决定计算什么时候应该停止。它只需要调用request_stop()，同时其他线程的should_stop()将会返回True，然后都停下来

```python
# 线程体：循环执行，直到Coordinator收到了停止请求。
 # 如果某些条件为真，请求Coordinator去停止其他线程。
def MyLoop(coord):
    while not coord.should_stop():
        ...do something...
        if ...some condition...:
            coord.request_stop()

# Main code: create a coordinator.
coord = Coordinator()

# Create 10 threads that run 'MyLoop()'
threads = [threading.Thread(target=MyLoop, args=(coord)) for i in xrange(10)]

# Start the threads and wait for all of them to stop. for t in threads: t.start()
    coord.join(threads)
```
这只是一个简单的例子，具体实现的时候比较灵活。实际中Coordinator会置入queue中，负责在得到线程关闭的请求后，关闭queue启动的多个线程。

### 队列管理器QueueRunner
创建并启动单个队列管理器的多个线程

```python
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
coord.request_stop()
coord.join(enqueue_threads)
```
- When you later call the create_threads() method, the QueueRunner will create one thread for each op in enqueue_ops. Each thread will run its enqueue op in parallel with the other threads.
- If a coordinator is given, this method starts an additional thread to close the queue when the coordinator requests a stop or exception error.

### 异常处理
- 通过queue runners启动的线程不仅仅只处理推送样本到队列。他们还捕捉和处理由队列产生的异常，包括OutOfRangeError异常，这个异常是用于报告队列被关闭
- 使用Coordinator的训练程序在主循环中必须同时捕捉和报告异常
  下面是对上面训练循环的改进版本：
```python
try:
    for step in xrange(1000000):
        if coord.should_stop():
            break
        sess.run(train_op)
except Exception, e:
# Report exceptions to the coordinator.
    coord.request_stop(e)

# Terminate as usual.  It is innocuous to request stop twice.
coord.request_stop()
coord.join(threads)
```
----------------------------------
接下来把队列 多线程利用起来读取数据。
先看一个简单的例子：
```python
import tensorflow as tf
import numpy as np

# 生成一个从1到100的数列
data_en_q = np.linspace(1,100,100)
# 创建一个可容纳50个元素的先进先出的常规队列，其元素类型为tf.uint8
qr = tf.FIFOQueue(capacity=50, dtypes=[tf.uint8,], shapes=[[]])
# en_qr为压数据进队列操作
en_qr = qr.enqueue_many([data_en_q])
# de_qr_1每次拿60个数据
de_qr_1 = qr.dequeue_many(60)
# de_qr_2每次拿50个数据
de_qr_2 = qr.dequeue_many(50)

# 创建一个包含两个线程的队列管理器，用于处理入队操作
qr = tf.train.QueueRunner(qr, enqueue_ops=[en_qr] * 2)
# 创建一个协调器，用于协调不同线程
coord = tf.train.Coordinator()

with tf.Session() as sess:
    # 启动队列管理器管理的线程
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    data_1 = list()
    data_2 = list()
    # 主线程消费100次，每次根据出队要求获取不同的元素个数
    for index in range(100):
        # 判断主线程是否需要停止退出
        if coord.should_stop():
            break
        meta1,meta2 = sess.run([de_qr_1,de_qr_2])
        data_1.append(meta1)
        data_2.append(meta2)
    # 主线程完成拿数据操作，打印出保存的数据
    print(data_1)
    print(data_2)
    # 主线程已完成任务，请求关闭处理入队操作的两个线程
    coord.request_stop()
    # 主线程等待所有线程关闭完毕再进入下一步
    coord.join(enqueue_threads)
```
注释的很清晰了。
接下来看一个实际使用的demo：

```python
# coding=utf-8
import time
import tensorflow as tf

# We simulate some raw input data 
# (think about it as fetching some data from the file system)
# let's say: batches of 128 samples, each containing 1024 data points
x_input_data = tf.random_normal([128, 1024], mean=0, stddev=1)

# We build our small model: a basic two layers neural net with ReLU
with tf.variable_scope("queue"):
    q = tf.FIFOQueue(capacity=5, dtypes=tf.float32) # enqueue 5 batches
    # We use the "enqueue" operation so 1 element of the queue is the full batch
    enqueue_op = q.enqueue(x_input_data)
    numberOfThreads = 1
    qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
    tf.train.add_queue_runner(qr)
    input = q.dequeue() # It replaces our input placeholder
    # We can also compute y_true right into the graph now
    y_true = tf.cast(tf.reduce_sum(input, axis=1, keep_dims=True) > 0, tf.int32)

with tf.variable_scope('FullyConnected'):
    w = tf.get_variable('w', shape=[1024, 1024], initializer=tf.random_normal_initializer(stddev=1e-1))
    b = tf.get_variable('b', shape=[1024], initializer=tf.constant_initializer(0.1))
    z = tf.matmul(input, w) + b
    y = tf.nn.relu(z)

    w2 = tf.get_variable('w2', shape=[1024, 1], initializer=tf.random_normal_initializer(stddev=1e-1))
    b2 = tf.get_variable('b2', shape=[1], initializer=tf.constant_initializer(0.1))
    z = tf.matmul(y, w2) + b2

with tf.variable_scope('Loss'):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(None, tf.cast(y_true, tf.float32), z)
    loss_op = tf.reduce_mean(losses)

with tf.variable_scope('Accuracy'):
    y_pred = tf.cast(z > 0, tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")

# We add the training op ...
adam = tf.train.AdamOptimizer(1e-2)
train_op = adam.minimize(loss_op, name="train_op")

startTime = time.time()
with tf.Session() as sess:
    # ... init our variables, ...
    sess.run(tf.global_variables_initializer())

    # ... add the coordinator, ...
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # ... check the accuracy before training (without feed_dict!), ...
    sess.run(accuracy)

    # ... train ...
    # 此处使用try语句，其实是没有必要的，但是实际中使用tf.train.QueueRunner作为文件队列是设置了最大训练迭代数，
    #在文件队列的出队操作数大于"num_epoches*队列容量"时，从文件队列读取的操作会读到一个"EOF",
    #这样最后一个样本出队的操作会得到一个tf.OutOfRangeError的错误
    try:
        for i in range(5000):
            #  ... without sampling from Python and without a feed_dict !
            _, loss = sess.run([train_op, loss_op])

            # We regularly check the loss
            if i % 500 == 0:
                print('iter:%d - loss:%f' % (i, loss))
    except tf.errors.OutOfRangeError:
        print 'Done training -- epoch limit reached'
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
 
    coord.join(threads)
    
    # Finally, we check our final accuracy
    sess.run(accuracy)

print("Time taken: %f" % (time.time() - startTime))
```