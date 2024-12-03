#!/usr/bin/env bash

# 检测当前使用的 shell
current_shell=$(ps -p $$ -ocomm=)

read_proxy() {
    read -p "如果需要使用代理，请输入代理地址（格式：IP:端口），否则直接按回车跳过: " proxy
}

set_bash_proxy() {
    if [[ ! -z "$proxy" ]]; then
        export http_proxy="http://$proxy"
        export https_proxy="https://$proxy"
        export ftp_proxy="ftp://$proxy"
        export socks_proxy="socks://$proxy"
        export no_proxy="localhost,127.0.0.1,::1"
        echo "已设置代理为 http://$proxy"
        echo "已设置代理为 https://$proxy"
        echo "已设置代理为 ftp://$proxy"
        echo "已设置代理为 socks://$proxy"
        echo "已设置 no_proxy 为 localhost,127.0.0.1,::1"
    else
        echo "未设置代理，继续执行..."
    fi
}

set_fish_proxy() {
    read -P "如果需要使用代理，请输入代理地址（格式：IP:端口），否则直接按回车跳过: " proxy
    if [ -n "$proxy" ]
        set -x http_proxy "http://$proxy"
        set -x https_proxy "https://$proxy"
        set -x ftp_proxy "ftp://$proxy"
        set -x socks_proxy "socks://$proxy"
        set -x no_proxy "localhost,127.0.0.1,::1"
        echo "已设置代理为 http://$proxy"
        echo "已设置代理为 https://$proxy"
        echo "已设置代理为 ftp://$proxy"
        echo "已设置代理为 socks://$proxy"
        echo "已设置 no_proxy 为 localhost,127.0.0.1,::1"
    else
        echo "未设置代理，继续执行..."
    end
}

if [[ "$current_shell" == "fish" ]]; then
    set_fish_proxy
else
    set_bash_proxy
fi
