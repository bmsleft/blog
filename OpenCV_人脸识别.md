## OpenCV人脸识别

### 目标方案

在树莓派上，PiCamera获得视频流，OpenCV识别， 然后用mjpg-streamer内发布http的视频流服务。

本篇文章主要记录OpenCV人脸识别实现的细节。

### OpenCV的人脸识别基础

人脸识别的技术门槛应该很低了，而且是AI技术中成熟度比较高的应用之一。现在很多好用的API，这里用到OpenCV的开源实现。

```Python
import cv2

face_patterns = cv2.CascadeClassifier('/usr/local/opt/opencv3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
sample_image = cv2.imread('/Users/abel/201612.jpg')
faces = face_patterns.detectMultiScale(sample_image,scaleFactor=1.1,minNeighbors=5,minSize=(100, 100))
for (x, y, w, h) in faces:
    cv2.rectangle(sample_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imwrite('/Users/abel/201612_detected.png', sample_image);
```

上述代码就是使用OpenCV识别一副图片中的多个人脸。代码来源于这个[blog](http://blog.csdn.net/wireless_com/article/details/64120516)， 参数是个比较重要的东西，要知道每个参数对应的效果。

### 视频流中的OpenCV人脸识别

这里是使用OpenCV的cascade识别人脸和眼睛，并用颜色框标识出来。OpenCV会自动调用电脑的摄像头，因此确保摄像头可用。

```python
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('D:\\bms\\workspace\\PythonProjects\\OpenCVDemo\\haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('D:\\bms\\workspace\\PythonProjects\\OpenCVDemo\\haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('frame', img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

这里说一下，haar_cascade_xxx.xml是个Haar分类器进行识别的参数。可以从[OpenCV的源码](https://github.com/opencv/opencv/tree/master/data/haarcascades)里找到基本的（人脸 眼睛之类）分类器参数。使用的时候需要下载到本地。

Haar分类器的原理之后再详细研究总结一下。

好了，现在将一开始的图片中检测人脸的代码优化一下，加入额外功能：

```Python
#!/usr/bin/python
'''
used for detect face on one image
Usage: python detect_face.py -i <inputfile> -o <outputfile>
'''
import cv2 as cv
import os, sys, getopt


def detect_face(infile, outfile):
    if not os.path.exists(infile):
        print("Please give input file!")
        return

    if not os.path.exists('haarcascade_frontalface_default.xml'):
        print("Please put haarcascade_frontalface_default.xml in current dir")
        return 
    
    infile_path = os.path.abspath(infile)
    image = cv.imread(infile_path, 1)
    gray = cv.imread(infile_path, 0)
    haarcascade_frontalface_path = os.path.abspath('haarcascade_frontalface_default.xml')
    face_cascade = cv.CascadeClassifier(haarcascade_frontalface_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv.imwrite(outfile, image)


def main(argv):
    infile = 'input.jpg'
    outfile = 'output.jpg'
    try:
        opts, args = getopt.getopt(argv, "hi:o:")
    except getopt.GetoptError:
        print("detect_face.py -i <inputfile> -o <outputfile>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("detect_face.py -i <inputfile> -o <outputfile>")
            sys.exit()
        elif opt == '-i':
            infile = arg
        elif opt == '-o':
            outfile = arg

    detect_face(infile, outfile)

if __name__ == "__main__":
    main(sys.argv[1:])
```

这样就可以使用shell调用人脸识别的功能了。

