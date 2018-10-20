### 引言

最近看图像处理相关的知识，涉及到颜色空间的转换操作比较多，总是感觉不是了解的很透彻，这里收集整理一下相关概念，做个记录。

### 概览

颜色空间（彩色模型、色彩空间、 彩色系统etc）是对色彩的一种描述方式，定义有很多种，区别在于面向不同的应用背景。

​    例如显示器中采用的RGB颜色空间是基于物体发光定义的（RGB正好对应光的三原色：Red，Green，Blue）；工业印刷中常用的CMY颜色空间是基于光反射定义的（CMY对应了绘画中的三原色：Cyan，Magenta，Yellow）；HSV、HSL两个颜色空间都是从人视觉的直观反映而提出来的（H是色调，S是饱和度，I是强度）。

### RGB颜色空间

颜色的加法混色原理，从黑色不断叠加Red，Green，Blue的颜色，最终可以得到白色光。如图：

![](http://img.blog.csdn.net/20140310091126703?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2Vpd2VpZ2ZrZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

将R、G、B三个通道作为笛卡尔坐标系中的X、Y、Z轴，就得到了一种对于颜色的空间描述。  在计算机中编程RGB每一个分量值都用8位（bit）表示，可以产生256*256*256=16777216中颜色，这就是经常所说的“24位真彩色”。

![](http://img.blog.csdn.net/20140310091844296?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2Vpd2VpZ2ZrZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)





### CMY颜色空间

另一种基于颜色减法混色原理的颜色模型。在工业印刷中它描述的是需要在白色介质上使用何种油墨，通过光的反射显示出颜色的模型。CMYK描述的是青，品红，黄和黑四种油墨的数值。

![](http://img.blog.csdn.net/20140310092305921?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2Vpd2VpZ2ZrZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



### HSV颜色空间

是根据颜色的直观特性由A. R. Smith在1978年创建的一种颜色空间, 也称六角锥体模型(Hexcone Model)。RGB和CMY颜色模型都是面向硬件的，而HSV（Hue Saturation Value）颜色模型是面向用户的。

这个模型中颜色的参数分别是：色调（H：hue），饱和度（S：saturation），亮度（V：value）。这是根据人观察色彩的生理特征而提出的颜色模型（人的视觉系统对亮度的敏感度要强于色彩值，这也是为什么计算机视觉中通常使用灰度即亮度图像来处理的原因之一）。
​    色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°。它们的补色是：黄色为60°，青色为180°,品红为300°；
​    饱和度S：取值范围为0.0～1.0；
​    亮度V：取值范围为0.0(黑色)～1.0(白色)。

![](http://img.blog.csdn.net/20140310092838734?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2Vpd2VpZ2ZrZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

![](http://img.blog.csdn.net/20140310092931359?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2Vpd2VpZ2ZrZA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



 HSV模型的三维表示从RGB立方体演化而来。设想从RGB沿立方体对角线的白色顶点向黑色顶点观察，就可以看到立方体的六边形外形。六边形边界表示色彩，水平轴表示纯度，明度沿垂直轴测量。与加法减法混色的术语相比，使用色相，饱和度等概念描述色彩更自然直观。

### HSL颜色空间

与HSV类似，只不过把V：Value替换为了L：Lightness。这两种表示在用目的上类似，但在方法上有区别。二者在数学上都是圆柱，但HSV（色相，饱和度，色调）在概念上可以被认为是颜色的倒[圆锥体](http://zh.wikipedia.org/wiki/%E5%9C%86%E9%94%A5%E4%BD%93)（黑点在下顶点，白色在上底面圆心），HSL在概念上表示了一个双圆锥体和圆球体（白色在上顶点，黑色在下顶点，最大横切面的圆心是半程灰色）。注意尽管在HSL和HSV中“色相”指称相同的性质，它们的“饱和度”的定义是明显不同的。对于一些人，HSL更好的反映了“饱和度”和“亮度”作为两个独立参数的直觉观念，但是对于另一些人，它的饱和度定义是错误的，因为非常柔和的几乎白色的颜色在HSL可以被定义为是完全饱和的。对于HSV还是HSL更适合于人类用户界面是有争议的。

### Lab颜色空间

是由CIE(国际照明委员会)制定的一种色彩模式。自然界中任何一点色都可以在Lab空间中表达出来，它的色彩空间比RGB空间还要大。另外，这种模式是以数字化方式来描述人的视觉感应， 与设备无关，所以它弥补了RGB和CMYK模式必须依赖于设备色彩特性的不足。 由于Lab的色彩空间要 比RGB模式和CMYK模式的色彩空间大。这就意味着RGB以及CMYK所能描述的色彩信息在Lab空间中都能得以影射。Lab颜色空间取坐标Lab，其中L亮度；a的正数代表红色，负端代表绿色；b的正数代表黄色，负端代表兰色。不像[RGB](http://zh.wikipedia.org/wiki/RGB)和[CMYK](http://zh.wikipedia.org/wiki/CMYK)色彩空间，Lab颜色被设计来接近人类视觉。它致力于感知均匀性，它的L分量密切匹配人类亮度感知。因此可以被用来通过修改a和b分量的输出[色阶](http://zh.wikipedia.org/w/index.php?title=%E8%89%B2%E9%98%B6&action=edit&redlink=1)来做精确的颜色平衡，或使用L分量来调整亮度对比。