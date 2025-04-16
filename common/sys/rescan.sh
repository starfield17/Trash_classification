#!/bin/bash
sleep 5
if ! lsmod | grep -q ch341; then
    sudo modprobe ch341
fi
USB_DEVICES=$(find /sys/bus/usb/devices -name "usb*" -type l)
for dev in $USB_DEVICES; do
    DEV_PATH=$(readlink -f $dev)
    if [ -e "$DEV_PATH/authorized" ]; then
        echo "rescan: $DEV_PATH"
        echo 0 | sudo tee "$DEV_PATH/authorized" > /dev/null
        sleep 1
        echo 1 | sudo tee "$DEV_PATH/authorized" > /dev/null
    fi
done
if [ ! -e /dev/ttyUSB0 ]; then
    echo "not detectd /dev/ttyUSB0，try to reload usb driver..."
    sudo rmmod ch341 2>/dev/null
    sleep 1
    sudo modprobe ch341
fi

echo "USB device scan completed"

