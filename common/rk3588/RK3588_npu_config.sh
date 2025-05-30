#!/bin/bash

# Exit if any command fails
set -e

# Update package list and install necessary dependencies
echo "Updating package list and installing dependencies..."
sudo apt-get update
sudo apt-get install -y cmake build-essential git python3-pip

# Configure proxy (optional)
read -p "Enter proxy address if needed (format: IP:Port), otherwise press Enter to skip: " proxy
if [[ ! -z "$proxy" ]]; then
    export http_proxy="http://$proxy"
    export https_proxy="https://$proxy"
    export ftp_proxy="ftp://$proxy"
    export socks_proxy="socks://$proxy"
    export no_proxy="localhost,127.0.0.1,::1"
    echo "Proxy set to http://$proxy"
    echo "Proxy set to https://$proxy"
    echo "Proxy set to ftp://$proxy"
    echo "Proxy set to socks://$proxy"
    echo "no_proxy set to localhost,127.0.0.1,::1"
else
    echo "No proxy set, continuing..."
fi

# Clone or update rknpu2 repository
if [ -d "rknpu2" ]; then
    echo "rknpu2 directory already exists, attempting to pull latest code..."
    cd rknpu2
    git pull
    cd ..
else
    echo "Cloning rknpu2 repository..."
    git clone https://github.com/rockchip-linux/rknpu2

    # Check if cloning was successful
    if [ $? -ne 0 ]; then
        echo "Failed to clone repository, please check network connection or repository address."
        exit 1
    fi
fi

# Copy library files to /usr/lib/
echo "Copying library files to /usr/lib/..."
if [ -f rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so ]; then
    sudo cp rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
else
    echo "File librknnrt.so does not exist, please check the path."
    exit 1
fi

if [ -f rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknn_api.so ]; then
    sudo cp rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknn_api.so /usr/lib/
else
    echo "File librknn_api.so does not exist, please check the path."
    exit 1
fi

# Copy header files to /usr/include/rknn/
echo "Copying header files to /usr/include/rknn/..."
sudo mkdir -p /usr/include/rknn

if [ -f rknpu2/runtime/RK3588/Linux/librknn_api/include/rknn_api.h ]; then
    sudo cp rknpu2/runtime/RK3588/Linux/librknn_api/include/rknn_api.h /usr/include/rknn/
else
    echo "File rknn_api.h does not exist, please check the path."
    exit 1
fi

if [ -f rknpu2/runtime/RK3588/Linux/librknn_api/include/rknn_matmul_api.h ]; then
    sudo cp rknpu2/runtime/RK3588/Linux/librknn_api/include/rknn_matmul_api.h /usr/include/rknn/
else
    echo "File rknn_matmul_api.h does not exist, please check the path."
    exit 1
fi

# Copy rknn_server related files to /usr/bin/
echo "Copying rknn_server related files to /usr/bin/..."
if [ -f rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/rknn_server ]; then
    sudo cp rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/rknn_server /usr/bin/
else
    echo "File rknn_server does not exist, please check the path."
    exit 1
fi

if [ -f rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/start_rknn.sh ]; then
    sudo cp rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/start_rknn.sh /usr/bin/
    sudo chmod +x /usr/bin/start_rknn.sh
else
    echo "File start_rknn.sh does not exist, please check the path."
    exit 1
fi

if [ -f rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/restart_rknn.sh ]; then
    sudo cp rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/restart_rknn.sh /usr/bin/
    sudo chmod +x /usr/bin/restart_rknn.sh
else
    echo "File restart_rknn.sh does not exist, please check the path."
    exit 1
fi

# Update linker configuration
echo "Updating linker configuration..."
sudo ldconfig

# Start rknn_server
echo "Starting rknn_server..."
sudo /usr/bin/start_rknn.sh

# Check if library files are correctly installed
echo "Checking if library files are correctly installed:"
ls -l /usr/lib/librknn*

# Check if rknn_server is running
echo "Checking if rknn_server is running:"
ps aux | grep rknn_server | grep -v grep

# Install rknn-toolkit2 using the official pip source
echo "Installing rknn-toolkit2..."
pip3 install rknn-toolkit2 -i https://pypi.org/simple

echo "Installation complete!"
