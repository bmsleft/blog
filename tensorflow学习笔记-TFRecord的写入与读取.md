------------
最近在学习tensorflow，关于tf自定义的数据记录格式tfrecord有必要了解一下。

因为在学习利用slim微调图像识别模型的时候，虽然tf准备好了模型已经retrain的脚本，但是数据是需要我们自己处理成tfrecord的格式的。这里转载一个我认为写的非常清晰的blog。
[why-every-tensorflow-developer-should-know-about-tfrecord](https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/ "why-every-tensorflow-developer-should-know-about-tfrecord")

------------

So, I suggest that the easier way to maintain a scalable architecture and a standard input format is to convert it into a tfrecord file.

Let me explain in terms of beginner’s language,

So when you are working with an image dataset, what is the first thing you do? Split into Train, Test, Validate sets, right? Also we will shuffle it to not have any biased data distribution if there are biased parameters like date.

Isn’t it tedious job to do the folder structure and then maintain the shuffle?

What if everything is in a single file and we can use that file to dynamically shuffle at random places and also change the ratio of train:test:validate from the whole dataset. Sounds like half the workload is removed right? A beginner’s nightmare of maintaining the different splits is now no more. This can be achieved by tfrecords.

Let’s see the difference between the code`Naive vs Tfrecord` 
### NAIVE
```python
import os 
import glob
import random

# Loading the location of all files - image dataset
# Considering our image dataset has apple or orange
# The images are named as apple01.jpg, apple02.jpg .. , orange01.jpg .. etc.
images = glob.glob('data/*.jpg')

# Shuffling the dataset to remove the bias - if present
random.shuffle(images)
# Creating Labels. Consider apple = 0 and orange = 1
labels = [ 0 if 'apple' in image else 1 for image in images ]
data = list(zip(images, labels))

# Ratio
data_size = len(data)
split_size = int(0.6 * data_size)

# Splitting the dataset
training_images, training_labels = zip(*data[:split_size])
testing_images, testing_labels = zip(*data[split_size:]
```

### TFRECORD
Follow the five steps and you are done with a single tfrecord file that holds all your data for proceeding.

Use tf.python_io.TFRecordWriter to open the tfrecord file and start writing.
Before writing into tfrecord file, the image data and label data should be converted into proper datatype. (byte, int, float)
Now the datatypes are converted into tf.train.Feature
Finally create an Example Protocol Buffer using tf.Example and use the converted features into it. Serialize the Example using serialize() function.
Write the serialized Example.

```python
import tensorflow as tf 
import numpy as np
import glob
from PIL import Image

# Converting the values into features
# _int64 is used for numeric values
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
tfrecord_filename = 'something.tfrecords'

# Initiating the writer and creating the tfrecords file.
writer = tf.python_io.TFRecordWriter(tfrecord_filename)

# Loading the location of all files - image dataset
# Considering our image dataset has apple or orange
# The images are named as apple01.jpg, apple02.jpg .. , orange01.jpg .. etc.
images = glob.glob('data/*.jpg')
for image in images[:1]:
  img = Image.open(image)
  img = np.array(img.resize((32,32)))
label = 0 if 'apple' in image else 1
feature = { 'label': _int64_feature(label),
              'image': _bytes_feature(img.tostring()) }

# Create an example protocol buffer
 example = tf.train.Example(features=tf.train.Features(feature=feature))
# Writing the serialized example.
 writer.write(example.SerializeToString())
writer.close()
```

If you closely see the process involved, it’s very simple.
`Data -> FeatureSet -> Example -> Serialized Example -> tfRecord.`
So to read it back, the process is reversed.
`tfRecord -> SerializedExample -> Example -> FeatureSet -> Data`

### READING FROM TFRECORD

```python
import tensorflow as tf 
import glob
reader = tf.TFRecordReader()
filenames = glob.glob('*.tfrecords')
filename_queue = tf.train.string_input_producer(
   filenames)
_, serialized_example = reader.read(filename_queue)
feature_set = { 'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)
           }
           
features = tf.parse_single_example( serialized_example, features= feature_set )
label = features['label']
 
with tf.Session() as sess:
  print sess.run([image,label])
```

You can also shuffle the files using the `tf.train.shuffle_batch()`

------------