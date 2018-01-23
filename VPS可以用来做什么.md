## VPS可以用来做什么
[TOC]

### VPS是什么

VPS(**Virtual private server**), **虚拟专用服务器**，是将一台服务器分区成多个虚拟专享服务器的服务。每个VPS可配置独立IP、内存、CPU资源、操作系统。

### VPS通常用来做什么

VPS的价值主要在于它有一个独立的固定的IP，这样就可以把资源放在上边提供服务让别人来访问了。目前VPS的个人用途主要有这么几种：

- 搭建个人服务器(邮箱、文件之类)
- 搭建个人网站
- 搭梯子翻~墙~~
- 跑跑一些demo啥的，骚气的操作很多，比如这里 [VPS有什么有趣的用途](https://www.zhihu.com/question/24284566)

### VPS运营商

国外的VPS运行商很多了。比较流行的就是这么几个：**Vultr、Digital Ocean、Linode、搬瓦工（bandwagonhost）**。 每个运营商旗下都有不同价位的产品，按需购买，一般乞丐版基本在5美刀/月。不过搬瓦工的更便宜哦，我自用的是20刀/年的洛杉矶直连中国的线路，目前搭建了个人博客，FTP文件服务，shadowsocks。 足够用了。

至于速度， 主要看两点：

1. **机房位置**

   一般都会选择洛杉矶机房和日本，比较近嘛

2. **本地接入宽带的运行商**

   这个就不好说了，主要靠实测，可以网上搜索各个运营商提供的IP地址ping下看看时间。能在200ms以内都算不错的。

下面记录个人自用的服务搭建过程

### 搭建Shadowsocks

作为搞技术的，经常用到Google，所以被~墙~的体验实在是很差劲。VPN经常被封，还好有Shadowsocks。

梯子的原理，可以参考 [shadowsocks.org](https://shadowsocks.org/en/index.html)， 和[wiki](https://zh.wikipedia.org/wiki/Shadowsocks)

搭建的过程很简单，网上各种教程满天飞，推荐一个比较详细的： [搬瓦工+shadowsocks：搭建自己的VPN服务器](https://moshuqi.github.io/2017/07/20/%E8%87%AA%E5%B7%B1%E6%90%AD%E5%BB%BAVPN%E6%9C%8D%E5%8A%A1%E5%99%A8/)

不过这个帖子里用的是VPS自带的shadowsocks安装服务，不足之处就是只能提供一个账号。想使用多账号的，建议不要使用自带安装，参考这里： [搬瓦工shadowsocks多用户配置教程](http://blog.csdn.net/d3soft/article/details/69663955)

### VPS加速

如果感觉梯子慢，可以去折腾一下加速。现在流行的加速都是基于BBR（Google提出的一个开源TCP拥塞控制算法）。如何加速，要看VPS是什么架构的。如果是KVM，可以比较容易开启BBR加速，OpenVZ的要麻烦一些。 贴个教程链接： [搬瓦工 VPS 安装并开启 Google BBR 教程（KVM / OpenVZ）](http://www.bandwagonhost.net/bandwagonhost-bbr.html)

### 搭建FTP文件服务

这个更简单了，只需要安装vsftpd就够用了。参考教程：[Centos7安装vsftpd (FTP服务器)](https://www.jianshu.com/p/9abad055fff6)。

### 搭建个人博客

这个完整的步骤如下：

1. 购买域名和VPS，并关联
2. 安装网站运行环境，一般都是LNMP或者LAMP的
3. 安装WordPress
4. 个性化配置个人博客

主要是这些，详细的教程见这里[VPS+LNMP+WordPress搭建个人网站/博客](http://jwcyber.com/build-site/)]

### WordPress的实用插件

如今的WordPress插件很丰富，可以帮助你快速配置各种功能。节省精力，专注到自己感兴趣的地方上。 推荐几个：

- 备份插件 [**BackWPup**](https://lighti.me/1727.html)
- 分享插件**Open Social**
- MarkDown写作插件**WP Editor.md**
- 免受垃圾评论**Akismet Anti-Spam**


