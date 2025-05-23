#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BLUE='\033[0;34m'

# Check if running with root privileges
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}Please run this script as root${NC}"
        echo "Usage: sudo $0"
        exit 1
    fi
}

# Scan and display available serial ports
scan_uart() {
    echo -e "${BLUE}=== Current serial ports in the system ===${NC}"
    echo -e "${YELLOW}Physical serial ports:${NC}"
    ls /dev/ttyAMA* /dev/ttyS* 2>/dev/null | while read port; do
        if [ -e "$port" ]; then
            echo -e "${GREEN}$port${NC}"
            # Try to get current port configuration
            stty -F $port -a 2>/dev/null | grep speed
        fi
    done

    echo -e "\n${YELLOW}USB serial devices:${NC}"
    ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | while read port; do
        if [ -e "$port" ]; then
            echo -e "${GREEN}$port${NC}"
            stty -F $port -a 2>/dev/null | grep speed
        fi
    done
}

# Display GPIO pin status
show_gpio_status() {
    echo -e "\n${BLUE}=== GPIO Pin Status ===${NC}"
    if command -v raspi-gpio >/dev/null; then
        raspi-gpio get | grep -E "GPIO|UART"
    else
        echo -e "${YELLOW}Note: Install raspi-gpio for more detailed GPIO status${NC}"
        echo "sudo apt-get install raspi-gpio"
    fi
}

# Display current UART configuration
show_uart_config() {
    echo -e "\n${BLUE}=== Current UART Configuration ===${NC}"
    if [ -f /boot/config.txt ]; then
        echo -e "${YELLOW}UART configuration detected in /boot/config.txt:${NC}"
        grep -i "uart" /boot/config.txt || echo "No UART-related configuration found"
        echo -e "\n${YELLOW}Available UART device tree overlays:${NC}"
        dtoverlay -h uart1
        dtoverlay -h uart2
    else
        echo -e "${RED}Cannot access /boot/config.txt${NC}"
    fi
}

# Configure new UART
configure_new_uart() {
    echo -e "\n${BLUE}=== Configure New UART ===${NC}"
    echo -e "Select UART to configure:"
    echo "1) UART1"
    echo "2) UART2"
    echo "3) Exit"
    read -p "Enter choice (1-3): " choice

    case $choice in
        1)
            configure_uart1
            ;;
        2)
            configure_uart2
            ;;
        3)
            echo "Exiting configuration"
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac
}

# Configure UART1
configure_uart1() {
    echo -e "\n${YELLOW}Configuring UART1${NC}"
    echo "Available TX pins: 14, 32, 40"
    echo "Available RX pins: 15, 33, 41"
    
    read -p "Enter TX pin number (default 14): " tx_pin
    read -p "Enter RX pin number (default 15): " rx_pin
    
    tx_pin=${tx_pin:-14}
    rx_pin=${rx_pin:-15}
    
    # Validate pin numbers
    if [[ ! "$tx_pin" =~ ^(14|32|40)$ ]] || [[ ! "$rx_pin" =~ ^(15|33|41)$ ]]; then
        echo -e "${RED}Error: Invalid pin configuration${NC}"
        return 1
    fi
    
    # Backup config file
    cp /boot/config.txt /boot/config.txt.backup
    
    # Add or update UART1 configuration
    if grep -q "dtoverlay=uart1" /boot/config.txt; then
        # Update existing configuration
        sed -i "/dtoverlay=uart1/c\dtoverlay=uart1,txd1_pin=$tx_pin,rxd1_pin=$rx_pin" /boot/config.txt
    else
        # Add new configuration
        echo "dtoverlay=uart1,txd1_pin=$tx_pin,rxd1_pin=$rx_pin" >> /boot/config.txt
    fi
    
    echo -e "${GREEN}UART1 configuration updated. Reboot required for changes to take effect.${NC}"
    echo "Configuration backup saved to /boot/config.txt.backup"
}

# Configure UART2
configure_uart2() {
    echo -e "\n${YELLOW}Configuring UART2${NC}"
    read -p "Enable CTS/RTS? (y/n): " enable_ctsrts
    
    # Backup config file
    cp /boot/config.txt /boot/config.txt.backup
    
    # Add or update UART2 configuration
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
    
    echo -e "${GREEN}UART2 configuration updated. Reboot required for changes to take effect.${NC}"
    echo "Configuration backup saved to /boot/config.txt.backup"
}

# Main menu
main_menu() {
    while true; do
        echo -e "\n${BLUE}=== Raspberry Pi UART Management Tool ===${NC}"
        echo "1) Scan current serial ports"
        echo "2) Show GPIO status"
        echo "3) Show UART configuration"
        echo "4) Configure new UART"
        echo "5) Exit"
        
        read -p "Select operation (1-5): " option
        
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
                echo -e "${GREEN}Thank you for using this tool!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option, please try again${NC}"
                ;;
        esac
    done
}

# Script entry point
check_root
main_menu
