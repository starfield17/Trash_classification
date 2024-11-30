read -p "如果需要使用代理，请输入代理地址（格式：IP:端口），否则直接按回车跳过: " proxy
if [[ ! -z "$proxy" ]]; then
    export http_proxy="http://$proxy"
    export https_proxy="https://$proxy"
    echo "已设置代理为 http://$proxy"
else
    echo "未设置代理，继续执行..."
fi
