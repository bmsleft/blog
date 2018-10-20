在之前的人脸识别方案中，主要使用OpenCV的Haar特征进行识别。不过就实际效果来看，不仅识别率有些低，而且速度慢一些。

最近发现了dlib（ http://dlib.net/）这个东西，很nice的一个开源C++库，实现了很多机器学习的算法，其中人脸识别的实现是依据《one millisecond face alignment with an ensemble of regression trees》这篇巨牛的论文。在dlib中完全可以再现论文中的效果，可以说很happy了。而且，dlib还提供了python的API，简直不能更友好啊~~~

然后，发现了这个 https://github.com/ageitgey/face_recognition 。依据dlib实现的人脸识别实用接口。可以完成多人脸识别以及标定。有时间的话搞到树莓派上玩玩。

不过有些担心树莓派的处理性能（如果配个屏幕的话就好了）。 感觉利用这个可以搞简易版的刷脸打卡。