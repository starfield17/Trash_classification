#!/bin/bash

# 如果脚本遇到错误则退出
set -e

# 更新包列表并安装必要的依赖
echo "更新包列表并安装依赖..."
sudo apt update
sudo apt install -y cmake build-essential git python3-pip
# 配置代理（可选）
read -p "如果需要使用代理，请输入代理地址（格式：IP:端口），否则直接按回车跳过: " proxy
if [[ ! -z "$proxy" ]]; then
    export http_proxy="http://$proxy"
    export https_proxy="https://$proxy"
    echo "已设置代理为 http://$proxy"
else
    echo "未设置代理，继续执行..."
fi

# 克隆 rknpu2 仓库
echo "克隆 rknpu2 仓库..."
git clone https://github.com/rockchip-linux/rknpu2
cd rknpu2/runtime/RK3588

# 复制库文件到 /usr/lib/
echo "复制库文件到 /usr/lib/..."
sudo cp Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
sudo cp Linux/librknn_api/aarch64/librknn_api.so /usr/lib/

# 复制头文件到 /usr/include/rknn/
echo "复制头文件到 /usr/include/rknn/..."
sudo mkdir -p /usr/include/rknn
sudo cp Linux/librknn_api/include/rknn_api.h /usr/include/rknn/
sudo cp Linux/librknn_api/include/rknn_matmul_api.h /usr/include/rknn/

# 复制 rknn_server 相关文件到 /usr/bin/
echo "复制 rknn_server 相关文件到 /usr/bin/..."
sudo cp Linux/rknn_server/aarch64/usr/bin/rknn_server /usr/bin/
sudo cp Linux/rknn_server/aarch64/usr/bin/start_rknn.sh /usr/bin/
sudo cp Linux/rknn_server/aarch64/usr/bin/restart_rknn.sh /usr/bin/

# 更新链接器配置
echo "更新链接器配置..."
sudo ldconfig

# 启动 rknn_server
echo "启动 rknn_server..."
sudo start_rknn.sh

# 检查库文件是否正确安装
echo "检查库文件是否正确安装："
ls -l /usr/lib/librknn*

# 检查 rknn_server 是否正在运行
echo "检查 rknn_server 是否正在运行："
ps aux | grep rknn_server | grep -v grep

# 安装 rknn-toolkit2，使用官方 pip 源
echo "安装 rknn-toolkit2..."
pip3 install rknn-toolkit2 -i https://pypi.org/simple

echo "安装完成！"
