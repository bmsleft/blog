

关于物体检测，目前深度学习网络的优势很明显，也是当前的热门应用。
目前，主要有三种主要的方法：

- [Faster R-CNNs](https://arxiv.org/abs/1506.01497) (Girshick et al., 2015)
- [You Only Look Once (YOLO)](https://arxiv.org/abs/1506.02640) (Redmon and Farhadi, 2015)
- [Single Shot Detectors (SSDs)](https://arxiv.org/abs/1512.02325) (Liu et al., 2015)

FR-CNNs慢但是准确性高，YOLO很快但是精度差点。SSDs平衡了这两种算法。如果再结合Google提出的MobileNets优化网络结构，可以得到很理想的效果。

下面贴出源码：

```python
import numpy as np
import cv2

imagefile = 'images/test02.jpg'
deploy_prototxt = 'MobileNetSSD_deploy.prototxt'
model = 'MobileNetSSD_deploy.caffemodel'
confidence_default = 0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[info] loading model...")
net = cv2.dnn.readNetFromCaffe(deploy_prototxt, model)

image = cv2.imread(imagefile)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

print("[info] computing object detections...")
net.setInput(blob)
detections = net.forward()

for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > confidence_default:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[info] {}".format(label))
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

cv2.imshow("output", image)
cv2.waitKey(0)
```

OpenCV需要3.3以上支持，因为代码中用到了cv2.dnn模块。

检测效果：

![output](https://s1.ax1x.com/2018/02/26/9wx0C4.png)

更多信息参考 [Object detection with deep learning and OpenCV](https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/)  