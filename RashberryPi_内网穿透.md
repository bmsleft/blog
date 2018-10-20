## RaspberryPi的内网穿透

> 所谓的内网穿透，就是能在公网上访问到你局域网里的机器。

内网穿透到工具有很多，最早接触到是花生壳，不过这个东西最近不厚道了，开始了收费。于是寻寻觅觅一些开源的工具。目前效果比较好的是这个frp（github地址[点这里](https://github.com/fatedier/frp/blob/master/README_zh.md)）

- frp的作用

> - 利用处于内网或防火墙后的机器，对外网环境提供 http 或 https 服务。
> - 对于 http, https 服务支持基于域名的虚拟主机，支持自定义域名绑定，使多个域名可以共用一个80端口。
> - 利用处于内网或防火墙后的机器，对外网环境提供 tcp 和 udp 服务，例如在家里通过 ssh 访问处于公司内网环境内的主机。

- 架构

  ![](https://github.com/fatedier/frp/blob/master/doc/pic/architecture.png?raw=true)

   所以说，看架构，你得先有个有公网IP点机器～～～
   还好还好，我们有VPS

## 使用示例

根据对应的操作系统及架构，从 [Release](https://github.com/fatedier/frp/releases) 页面下载最新版本的程序。

将 **frps** 及 **frps.ini** 放到具有公网 IP 的机器上。

将 **frpc** 及 **frpc.ini** 放到处于内网环境的机器上。

### 通过 ssh 访问公司内网机器

1. 修改 frps.ini 文件，这里使用了最简化的配置：

```
# frps.ini
[common]
bind_port = 7000
```

2. 启动 frps：

`./frps -c ./frps.ini`

3. 修改 frpc.ini 文件，假设 frps 所在服务器的公网 IP 为 x.x.x.x；

```
# frpc.ini
[common]
server_addr = x.x.x.x
server_port = 7000

[ssh]
type = tcp
local_ip = 127.0.0.1
local_port = 22
remote_port = 6000
```

4. 启动 frpc：

`./frpc -c ./frpc.ini`

5. 通过 ssh 访问内网机器，假设用户名为 test：

`ssh -oPort=6000 test@x.x.x.x`



好了，这样就能在公司ssh登陆到自己的树莓派玩了～～  

frp的其他配置使用可以继续折腾～～