### 基本方案

在树莓派上，PiCamera获得视频流，OpenCV识别后将文件写入指定目录， 然后用mjpg-streamer发布http的视频流服务。

此方案的实现比较简单，主要是将之前已经实现的功能聚合在一起。不过，缺点是明显的：帧率会很低，画面应该很迟钝。具体效果，等完成看看吧。

```Python
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from time import sleep
from picamera import PiCamera
import cv2 as cv


def main():
    camera = PiCamera()
    camera.start_preview()
    sleep(0.1)

    out_path = '/tmp/face_stream'
    if not os.path.exist(out_path):
        os.mkdir(out_path)

    outfile = out_path + '/face.jpg'
    for filename in camera.capture_continuous('img{timestamp:%Y-%m-%d-%H-%M}.jpg'):
        detect_face(filename, outfile)
        os.remove(filename)
        sleep(0.05)


def detect_face(infile, outfile):
    if not os.path.exists(infile):
        print("Please give input file!")
        return

    if not os.path.exists('haarcascade_frontalface_default.xml'):
        print("Please put haarcascade_frontalface_default.xml in current dir")
        return

    infile_path = os.path.abspath(infile)
    image = cv.imread(infile_path, 1)
    #gray = cv.imread(infile_path, 0)
    haarcascade_frontalface_path = os.path.abspath('haarcascade_frontalface_default.xml')
    face_cascade = cv.CascadeClassifier(haarcascade_frontalface_path)
    faces = face_cascade.detectMultiScale(image, 1.2, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv.imwrite(outfile, image)
   
 
if __name__ == "__main__":
    main()
```

后台运行上述代码后，接着启动mjpg streamer，指定输入流为上一步输出的标识人脸的图片。

```shell
MJPG_STREAMER_PATH="/home/pi/Downloads/sourcecode/mjpg-streamer/mjpg-streamer-experimental"
cd $MJPG_STREAMER_PATH
export LD_LIBRARY_PATH=.
mjpg_streamer -i "input_file.so -f /tmp/face_stream -n face.jpg" -o "output_http.so -w ./www"
```

按照之前[树莓派的实时网络视频监控](http://flyingcat.top/2018/01/raspberrypi-video-monitor/)中的shell脚本后台运行的方法，运行上述脚本后，可以通过http查看检测结果了。
相比较来说，我们仅仅更改了输入流，将摄像头的输入替换为了OpenCV识别后的图像流。

效果图：（不过不得不说，python真是慢啊，，，可以改用c++写一个

![](https://ws1.sinaimg.cn/large/006tNc79ly1fnyzho2xf9j312e166q5l.jpg)



### 利用mjpg_streamer的input_opencv 插件调用OpenCV的人脸识别功能

这个功能需要Python3.X版本，并且，OpenCV需要使用OpenCV3.1.0 + 。好吧，我还没有配置这个环境，也不确定这样做的性能改善如何。不过应该会有一些提升。这里是具体[usage](https://github.com/jacksonliam/mjpg-streamer/blob/master/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_py/README.md)

基本来说，启动服务时，这样：

`mjpg_streamer -i "input_opencv.so --filter cvfilter_py.so --fargs path/to/filter.py"`

自定义的filter.py需要这样：

 ```python
def filter_fn(img):
    '''
        :param img: A numpy array representing the input image
        :returns: A numpy array to send to the mjpg-streamer output plugin
    '''
    return img
    
def init_filter():
    return filter_fn
 ```

这样子我们就可以把OpenCV的人脸识别部分移到filter_fn()函数中。我还没有实现这个，有时间再搞搞。

