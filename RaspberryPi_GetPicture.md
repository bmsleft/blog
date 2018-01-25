## 树莓派的定时拍照并上传到百度云

###安装摄像头以及配置开关打开

如何连接以及打开摄像头功能可以参考官方[Document](https://www.raspberrypi.org/documentation/usage/camera/README.md)

###树莓派的拍照功能

树莓派自带针对camera的bash命令还是很实用的。官网有介绍如何使用。

- [raspistill](https://www.raspberrypi.org/documentation/usage/camera/raspicam/raspistill.md) 捕获静态图像


   raspistill -o cam.jpg
   raspistill -t 30000 -tl 2000 -o image%04d.jpg



- [raspivid](https://www.raspberrypi.org/documentation/usage/camera/raspicam/raspivid.md) 捕获视频 

  raspivid -o video.h264 -t 10000



### 使用picamera py库

如果想在Python中调用camera功能，推荐使用这个。

- 安装


  sudo apt-get update
  sudo apt-get install python-picamera

- 调用


```python
  import picamera
  camera = picamera.PiCamera()
  camera.capture('image.jpg')
```


   更多使用方法可以参考[documentation](http://picamera.readthedocs.org/)

### 百度云文件同步的Python模块ByPy

具体看github上的说明吧，很详细了。[ReadMe](https://github.com/houtianze/bypy)

### 使用picamera和bypy模块实现定时拍照并自动上传到百度云

直接贴代码吧，很easy。

```Python
import os
from time import sleep
from datetime import datetime, timedelta
from picamera import PiCamera
from bypy import ByPy

def wait(delay_minute = 1):
	next_time = (datetime.now() + timedelta(minutes=delay_minute)).replace(second=0, microsecond=0)
	delay = (next_time - datetime.now()).seconds
	sleep(delay)

by=ByPy()
camera = PiCamera()
camera.start_preview()
wait()

for filename in camera.capture_continuous('img{timestamp:%Y-%m-%d-%H-%M}.jpg'):
	print('capture %s' % filename)
	by.upload(filename)
	os.remove(filename)
	wait()
```

