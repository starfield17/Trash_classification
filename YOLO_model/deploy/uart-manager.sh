#!/bin/bash

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BLUE='\033[0;34m'

# 检查是否以root权限运行
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}请以root权限运行此脚本${NC}"
        echo "使用: sudo $0"
        exit 1
    fi
}

# 扫描并显示当前可用的串口
scan_uart() {
    echo -e "${BLUE}=== 当前系统中的串口设备 ===${NC}"
    echo -e "${YELLOW}物理串口:${NC}"
    ls /dev/ttyAMA* /dev/ttyS* 2>/dev/null | while read port; do
        if [ -e "$port" ]; then
            echo -e "${GREEN}$port${NC}"
            # 尝试获取串口的当前配置
            stty -F $port -a 2>/dev/null | grep speed
        fi
    done

    echo -e "\n${YELLOW}USB串口设备:${NC}"
    ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | while read port; do
        if [ -e "$port" ]; then
            echo -e "${GREEN}$port${NC}"
            stty -F $port -a 2>/dev/null | grep speed
        fi
    done
}

# 显示GPIO引脚状态
show_gpio_status() {
    echo -e "\n${BLUE}=== GPIO引脚状态 ===${NC}"
    if command -v raspi-gpio >/dev/null; then
        raspi-gpio get | grep -E "GPIO|UART"
    else
        echo -e "${YELLOW}提示: 安装raspi-gpio可以查看更详细的GPIO状态${NC}"
        echo "sudo apt-get install raspi-gpio"
    fi
}

# 显示当前UART配置
show_uart_config() {
    echo -e "\n${BLUE}=== 当前UART配置 ===${NC}"
    if [ -f /boot/config.txt ]; then
        echo -e "${YELLOW}从/boot/config.txt中检测到的UART配置:${NC}"
        grep -i "uart" /boot/config.txt || echo "未找到UART相关配置"
        echo -e "\n${YELLOW}可用的UART设备树覆盖:${NC}"
        dtoverlay -h uart1
        dtoverlay -h uart2
    else
        echo -e "${RED}无法访问/boot/config.txt${NC}"
    fi
}

# 配置新的UART
configure_new_uart() {
    echo -e "\n${BLUE}=== 配置新的UART ===${NC}"
    echo -e "选择要配置的UART:"
    echo "1) UART1"
    echo "2) UART2"
    echo "3) 退出"
    read -p "请选择 (1-3): " choice

    case $choice in
        1)
            configure_uart1
            ;;
        2)
            configure_uart2
            ;;
        3)
            echo "退出配置"
            ;;
        *)
            echo -e "${RED}无效的选择${NC}"
            ;;
    esac
}

# 配置UART1
configure_uart1() {
    echo -e "\n${YELLOW}配置UART1${NC}"
    echo "可用的TX引脚: 14, 32, 40"
    echo "可用的RX引脚: 15, 33, 41"
    
    read -p "输入TX引脚号 (默认14): " tx_pin
    read -p "输入RX引脚号 (默认15): " rx_pin
    
    tx_pin=${tx_pin:-14}
    rx_pin=${rx_pin:-15}
    
    # 验证输入的引脚是否有效
    if [[ ! "$tx_pin" =~ ^(14|32|40)$ ]] || [[ ! "$rx_pin" =~ ^(15|33|41)$ ]]; then
        echo -e "${RED}错误: 无效的引脚配置${NC}"
        return 1
    fi
    
    # 备份配置文件
    cp /boot/config.txt /boot/config.txt.backup
    
    # 添加或更新UART1配置
    if grep -q "dtoverlay=uart1" /boot/config.txt; then
        # 更新现有配置
        sed -i "/dtoverlay=uart1/c\dtoverlay=uart1,txd1_pin=$tx_pin,rxd1_pin=$rx_pin" /boot/config.txt
    else
        # 添加新配置
        echo "dtoverlay=uart1,txd1_pin=$tx_pin,rxd1_pin=$rx_pin" >> /boot/config.txt
    fi
    
    echo -e "${GREEN}UART1配置已更新。需要重启才能生效。${NC}"
    echo "配置备份已保存到 /boot/config.txt.backup"
}

# 配置UART2
configure_uart2() {
    echo -e "\n${YELLOW}配置UART2${NC}"
    read -p "是否启用CTS/RTS? (y/n): " enable_ctsrts
    
    # 备份配置文件
    cp /boot/config.txt /boot/config.txt.backup
    
    # 添加或更新UART2配置
    if [ "$enable_ctsrts" = "y" ]; then
        if grep -q "dtoverlay=uart2" /boot/config.txt; then
            sed -i "/dtoverlay=uart2/c\dtoverlay=uart2,ctsrts" /boot/config.txt
        else
            echo "dtoverlay=uart2,ctsrts" >> /boot/config.txt
        fi
    else
        if grep -q "dtoverlay=uart2" /boot/config.txt; then
            sed -i "/dtoverlay=uart2/c\dtoverlay=uart2" /boot/config.txt
        else
            echo "dtoverlay=uart2" >> /boot/config.txt
        fi
    fi
    
    echo -e "${GREEN}UART2配置已更新。需要重启才能生效。${NC}"
    echo "配置备份已保存到 /boot/config.txt.backup"
}

# 主菜单
main_menu() {
    while true; do
        echo -e "\n${BLUE}=== 树莓派串口管理工具 ===${NC}"
        echo "1) 扫描当前串口设备"
        echo "2) 显示GPIO状态"
        echo "3) 显示UART配置"
        echo "4) 配置新的UART"
        echo "5) 退出"
        
        read -p "请选择操作 (1-5): " option
        
        case $option in
            1)
                scan_uart
                ;;
            2)
                show_gpio_status
                ;;
            3)
                show_uart_config
                ;;
            4)
                configure_new_uart
                ;;
            5)
                echo -e "${GREEN}谢谢使用！${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}无效的选项，请重试${NC}"
                ;;
        esac
    done
}

# 脚本入口
check_root
main_menu
