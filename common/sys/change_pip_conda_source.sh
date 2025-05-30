#!/bin/bash

# Check if required package managers are installed
check_package_managers() {
    local missing_managers=()
    
    if ! command -v pip &> /dev/null; then
        missing_managers+=("pip")
    fi
    
    if ! command -v conda &> /dev/null; then
        missing_managers+=("conda")
    fi
    
    if [ ${#missing_managers[@]} -ne 0 ]; then
        echo "Warning: The following package managers are not installed:"
        for manager in "${missing_managers[@]}"; do
            echo "- $manager"
        done
        echo "Some features may not be available."
        echo "----------------------------------------"
    fi
}

# Define mirror URLs
declare -A PIP_SOURCES=(
    ["1"]="USTC (China) https://mirrors.ustc.edu.cn/pypi/web/simple"
    ["2"]="Tsinghua (China) https://pypi.tuna.tsinghua.edu.cn/simple"
    ["3"]="Tencent Cloud (China) http://mirrors.cloud.tencent.com/pypi/simple"
    ["4"]="Douban (China) http://pypi.douban.com/simple"
    ["5"]="Alibaba Cloud (China) https://mirrors.aliyun.com/pypi/simple/"
    ["6"]="Default source (Restore default) default"
)

declare -A CONDA_SOURCES=(
    ["1"]="Tsinghua (China) https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/"
    ["2"]="USTC (China) https://mirrors.ustc.edu.cn/anaconda/pkgs/main/"
    ["3"]="Official source https://repo.anaconda.com/pkgs/main/"
)

# Backup conda config file
backup_condarc() {
    if [ -f ~/.condarc ]; then
        cp ~/.condarc ~/.condarc.bak
        echo "Existing .condarc backed up to .condarc.bak"
    fi
}

# Show main menu
show_main_menu() {
    echo "========================================"
    echo "      Python Package Manager Source Switcher"
    echo "========================================"
    echo "Select package manager to configure:"
    echo "1) pip"
    echo "2) conda"
    echo "0) Exit"
    echo "========================================"
}

# Show pip source menu
show_pip_menu() {
    echo "========================================"
    echo "        pip Source Selection Menu"
    echo "========================================"
    echo "Select pip source to set:"
    for key in "${!PIP_SOURCES[@]}"; do
        echo "$key) ${PIP_SOURCES[$key]}"
    done
    echo "9) Back to main menu"
    echo "0) Exit"
    echo "========================================"
}

# Show conda source menu
show_conda_menu() {
    echo "========================================"
    echo "        conda Source Selection Menu"
    echo "========================================"
    echo "Select conda source to set:"
    for key in "${!CONDA_SOURCES[@]}"; do
        echo "$key) ${CONDA_SOURCES[$key]}"
    done
    echo "9) Back to main menu"
    echo "0) Exit"
    echo "========================================"
}

# Set pip source
set_pip_source() {
    local choice=$1
    local url

    case $choice in
        1)
            url="https://mirrors.ustc.edu.cn/pypi/web/simple"
            ;;
        2)
            url="https://pypi.tuna.tsinghua.edu.cn/simple"
            ;;
        3)
            url="http://mirrors.cloud.tencent.com/pypi/simple"
            ;;
        4)
            url="http://pypi.douban.com/simple"
            ;;
        5)
            url="https://mirrors.aliyun.com/pypi/simple/"
            ;;
        6)
            pip config unset global.index-url
            echo "----------------------------------------"
            echo "Restored default source settings"
            echo "----------------------------------------"
            return 0
            ;;
        9)
            return 1
            ;;
        0)
            echo "Thank you for using this tool!"
            exit 0
            ;;
        *)
            echo "Invalid choice, please try again."
            return 2
            ;;
    esac

    if [ $? -eq 0 ] && [ -n "$url" ]; then
        pip config set global.index-url "$url"

        echo "----------------------------------------"
        echo "Set pip source to: $url"
        echo "Current pip configuration:"
        pip config get global.index-url
        echo "----------------------------------------"
    fi
}

# Set conda source
set_conda_source() {
    local choice=$1
    local url
    local name

    case $choice in
        1)
            url="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/"
            name="Tsinghua (China)"
            ;;
        2)
            url="https://mirrors.ustc.edu.cn/anaconda/pkgs/main/"
            name="USTC (China)"
            ;;
        3)
            url="https://repo.anaconda.com/pkgs/main/"
            name="Official source"
            ;;
        9)
            return 1
            ;;
        0)
            echo "Thank you for using this tool!"
            exit 0
            ;;
        *)
            echo "Invalid choice, please try again."
            return 2
            ;;
    esac

    if [ $? -eq 0 ] && [ -n "$url" ]; then
        backup_condarc
        conda config --remove-key channels
        conda config --add channels "$url"
        conda config --set show_channel_urls yes

        echo "----------------------------------------"
        echo "Set Conda source to: $name - $url"
        echo "Current Conda configuration:"
        cat ~/.condarc
        echo "----------------------------------------"
    fi
}

# Main function
main() {
    check_package_managers

    while true; do
        show_main_menu
        read -p "Enter your choice [0-2]: " main_choice

        case $main_choice in
            1)  # pip configuration
                while true; do
                    show_pip_menu
                    read -p "Enter your choice [0-9]: " pip_choice
                    set_pip_source "$pip_choice"
                    if [ $? -eq 1 ]; then
                        break
                    fi
                done
                ;;
            2)  # conda configuration
                while true; do
                    show_conda_menu
                    read -p "Enter your choice [0-9]: " conda_choice
                    set_conda_source "$conda_choice"
                    if [ $? -eq 1 ]; then
                        break
                    fi
                done
                ;;
            0)  # Exit
                echo "Thank you for using this tool!"
                exit 0
                ;;
            *)  # Invalid choice
                echo "Invalid choice, please try again."
                ;;
        esac
    done
}

# Run main function
main
