最近看了目标检测的相关论文,想来用tf实现一下. 正巧手头上有本《21个项目玩转深度学习》，第五章是目标检测的示例。于是按照demo跑了一下。
原始项目Github: [https://github.com/hzy46/Deep-Learning-21-Examples/tree/master/chapter_5](https://github.com/hzy46/Deep-Learning-21-Examples/tree/master/chapter_5)

简单记录一下实现过程：
1. 安装TensorFlow Object Detection API 
    https://github.com/tensorflow/models/tree/master/research/object_detection 

2. 训练新的模型
      先在地址[http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 下载VOC2012数据集并解压。
    在项目的object_detection文件夹中新建voc目录，并将解压后的数据集拷贝进来，最终形成的目录为：
```
research/
  object_detection/
    voc/
      VOCdevkit/
        VOC2012/
          JPEGImages/
            2007_000027.jpg
            2007_000032.jpg
            2007_000033.jpg
            2007_000039.jpg
            2007_000042.jpg
            ………………
          Annotations/
            2007_000027.xml
            2007_000032.xml
            2007_000033.xml
            2007_000039.xml
            2007_000042.xml
            ………………
          ………………
```
然后制作数据的tfrecord格式
```
python create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=train --output_path=voc/pascal_train.record
python create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=val --output_path=voc/pascal_val.record
```
并且把label数据拷贝过来
``` cp data/pascal_label_map.pbtxt voc/ ```
下载模型文件[http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz) 并解压，解压后得到frozen_inference_graph.pb 、graph.pbtxt 、model.ckpt.data-00000-of-00001 、model.ckpt.index、model.ckpt.meta 5 个文件。在voc文件夹中新建一个 pretrained 文件夹，并将这5个文件复制进去。

复制一份config文件：

```
cp samples/configs/faster_rcnn_inception_resnet_v2_atrous_pets.config \
  voc/voc.config
```

并在voc/voc.config中修改7处需要重新配置的地方：
- num_class 修改为20.这是VOC2012中物体类别数。
- eval_config中的num_examples 改为5823， 这是验证集中的图片数目
- PATH_TO_BE_CONFIGURED 有5处，修改为自己的数目目录对应的地方。

训练模型的命令：
```
python train.py --train_dir voc/train_dir/ --pipeline_config_path voc/voc.config
```
然后就可是训练了，微调的模型相对来说比较快些。

使用TensorBoard：
```
tensorboard --logdir voc/train_dir/
```
3. 导出模型

运行(需要根据voc/train_dir/里实际保存的checkpoint，将1582改为训练时候保存的数值)：

python export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path voc/voc.config \
  --trained_checkpoint_prefix voc/train_dir/model.ckpt-1582
  --output_directory voc/export/
导出的模型是voc/export/frozen_inference_graph.pb 文件。

4. 预测单张图片
  可以根据object_detection下的object_detection_tutorial.ipynb编写自己的推断程序。
  我这里使用自己训练好的模型来预测：
```python
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join('voc/export') + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('voc', 'pascal_label_map.pbtxt')

NUM_CLASSES = 20

detection_graph = tf.Graph()

with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

 # For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
      print('now inference image:' + image_path)
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)

    plt.show()
```

好了，这样就能预测自己的数据了。
总的来说，使用的是tf已经预先编制好的代码来进行自己数据的微调与应用。这样子的话比较快的上手，不至于深究其中的细节。

至于算法代码的具体实现，有需要再写个blog详细记录一下。