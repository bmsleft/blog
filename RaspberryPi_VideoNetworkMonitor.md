

## 树莓派的实时网络视频监控

### 安装摄像头以及配置开关打开

如何连接以及打开摄像头功能可以参考官方[Document](https://www.raspberrypi.org/documentation/usage/camera/README.md)

### 实时网络视频监控方案

Google了一下现有的快速构建（就是拿来用用 呵呵）技术方案， 基本是是使用motion或者mjpg-streamer两种二选一：

- motion使用起来实在是简单，安装个包之后直接调用就好，不过延迟实在是厉害，网上吐槽声音一大片，暂时没找到很好的解决方法。

- mjpg-streamer ,顾名思义，是使用jpeg作为视频格式，[GitHub](https://github.com/jacksonliam/mjpg-streamer)上的说明是这样的：

  > mjpg-streamer is a command line application that copies JPEG frames from one or more input plugins to multiple output plugins. It can be used to stream JPEG files over an IP-based network from a webcam to various types of viewers such as Chrome, Firefox, Cambozola, VLC, mplayer, and other software capable of receiving MJPG streams.

  好处么，就是延迟小喽，不过要自己编译一下，很简单，跟着GitHub上的说明一步步来就好。

  ### 实现步骤

1. GitHub上下载最新源码。直接下载也行，配置了git的话比较方便

   `git clone https://github.com/jacksonliam/mjpg-streamer.git`

2. 编译准备

   ` sudo apt-get install cmake libjpeg8-dev `

3. 编译安装

   `cd mjpg-streamer-experimental`

    `make`

    `sudo make install`

4. 然后就可以直接使用了。。（真真是。。不能再easy了）

   `export LD_LIBRARY_PATH=.`

   `./mjpg_streamer -o "output_http.so -w ./www" -i "input_raspicam.so"`

5. 然后浏览器中手动输入如下地址就可以看到视频流了。

![运行之后的效果](https://ws1.sinaimg.cn/large/006tNc79ly1fns31i5zkpj317i1cy77l.jpg)



6. 优化一下， 写个脚本[demo2_mjpg_streamer.sh](https://github.com/bmsleft/RaspberryPiDemo/blob/master/demo2_mjpg_streamer.sh)，放在后台运行

   ` !/bin/bash `
   `MJPG_STREAMER_PATH="/home/pi/Downloads/sourcecode/mjpg-streamer/mjpg-streamer-experimental"`
   `cd $MJPG_STREAMER_PATH`
   `export LD_LIBRARY_PATH=.`
   `./mjpg_streamer -o "output_http.so -w ./www" -i "input_raspicam.so"`

   然后， `chmod +x demo2_mjpg_streamer.sh` `bash ./demo2_mjpg_streamer.sh >/dev/null 2>&1 &`

   好了，这时候就可以干别的了， 浏览器中仍然可以看到视频流。

   如果需要开启动，把`/usr/bin/bash ./demo2_mjpg_streamer.sh >/dev/null 2>&1 &` 加到 /etc/rc.local 中exit 0之前就好

  至于进一步分析这个视频，怎么利用视频流数据，这个之后有时间再研究。