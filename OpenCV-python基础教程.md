在传统的计算机视觉处理中，OpenCV依然是最为基础和常用的工具。 即便是如今深度学习如火如荼，在图像处理领域表现出越来越强大的潜力时，OpenCV也不失为预处理的有效手段。另外，OpenCV的工具箱里，也不断再融入机器学习的算法，充实自身才能应对变化。

之前用OpenCV的人脸识别功能以及视频处理功能，都是简单的使用而已。最近一点时间系统看了下教程，这里先贴出个教程链接以备忘：[OpenCV-Python中文教程](https://www.kancloud.cn/aollo/aolloopencv)

说明一下，我一向不习惯做教程，觉得浪费时间，东西都在网上，拿过来看就好了。不过总是要留一个系统的结构图的，以便追踪知识点的覆盖情况。毕竟知识这么多，能学到的仅仅是九牛一毛而已。

总结一下主要的概念点：

- 图像读取显示以及保存
  1. cv2.imread()
  2. cv2.imshow()
  3. cv2.imwrite()
- 视频操作
  1. cv2.VideoCapture() 
- 绘图函数
  1. cv2.line() , cv2.cicle() , cv2.rectangle() , cv2.ellipse() , cv2.putText()
- 图像基本操作图像可以看作是numpy的一个矩阵，矩阵操作可以看出图像操作：img.shape img.size img.dtyper,g,b=cv2.split(img)#拆分 img=cv2.merge(r,g,b)#合并

- 图像算术运算cv2.add() ,cv2.addWeighted() cv2.bitwise_and 
- 颜色空间转换cv2.cvtColor(input_imageﬂag) cv2.COLOR_BGR2GRAY cv2.COLOR_BGR2HSV
- 几何变换cv2.resize() cv2.warpAffine() cv2.getAffineTransForm() cv2.getPerspectiveTransform() cv2.warpPerspective
- 图像平滑
  1. 2D卷积 低通滤波（LPF）和高通滤波（HPF）。LPF用于去除噪音，模糊图像，HPF用于找到图像的边缘
  2. 平均 cv2.blur()和cv2.boxFilter()
  3. 高斯模糊 cv2.GaussianBlur()
  4. 中值模糊 cv2.medianBlur
  5. 双边滤波 cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
- 形态学转换
  1. 腐蚀 把前景物体的边界腐蚀掉，但是前景仍然是白色的。卷积核沿着图像滑动，如果与卷积核对应的原图像的所有像素值都是1，那么中心元素就保持原来的像素值，否则就变为零。 cv2.erode()
  2. 膨胀 与腐蚀相反，与卷积核对应的原图像的像素值中只要有一个是1，中心元素的像素值就是1。所以这个操作会增加图像中白色区域。cv2.dilation()
  3. 开运算 先进行腐蚀再进行膨胀就叫做开运算。被用来去除噪音。opening = cv2.morphotogyEx(img,cv2.MORPH_OPEN,kernel)
  4. 闭运算 先膨胀再腐蚀。被用来填充前景物体中的小洞，或者前景上的小黑点。 cv2.morphotogyEx(img,cv2.MORPH_CLOSE,kernel)
  5. 形态学梯度 就是一幅图像膨胀与腐蚀的差别，结果看上去就像前景物体的轮廓。gradient = cv2.morphotogyEx(img,cv2.MORPH_GRADIENT,kernel)
  6. 礼帽 原始图像与进行开运算之后得到的图像的差。 tophat = cv2.morphotogyEx(img,cv2.MORPH_TOPHAT,kernel)
  7. 黑帽 进行闭运算之后得到的图像与原始图像的差。 blackhat = cv2.morphotogyEx(img,cv2.MORPH_BLACKHAT,kernel)
- 图像梯度梯度就是求导。OpenCV提供了三种不同的梯度滤波器，或者说高通滤波器：Sobel，Scharr和Laplacian。Sobel和Scharr是求一阶或二阶导数。Scharr是对Sobel（使用小的卷积核求解梯度角度时）的优化，Laplacian是求二阶导数。Sobel算子是高斯平滑与微分操作的结合体，它的抗噪音能力很好。拉普拉斯算子可以使用二阶导数的形式定义，可假设其离散实现类似于二阶Sobel导数，事实上OpenCV在计算拉普拉斯算子时直接调用Sobel算子。
- Canny边缘检测处理流程： 噪音去除 > 计算图像梯度 > 非极大值抑制 > 滞后阀值 cv2.Canny(img,minVal, maxVal)
- 图像金字塔
  1. 高斯金字塔的顶部是通过将底部图像中的连续的行和列去除得到的。顶部图像中的每个像素值等于下一层图像中5个像素的高斯加权平均值。这样操作一次一个MxN的图像就变成了一个M/2xN/2的图像。所以这幅图像的面积就变为原来图像面积的四分之一。这被称为Octave。连续这样的操作，我们就会得到一个分辨率不断下降的图像金字塔。可以使用函数cv2.pyrDown()和cv2.pyrUp()构建图像金字塔。
  2. 拉普拉斯金字塔可以由高斯金字塔计算得来。图像看起来就像是边界图，其中很多像素都是0，常被用在图像压缩中。