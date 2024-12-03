#!/bin/bash

# 检查必要的包管理器是否安装
check_package_managers() {
    local missing_managers=()
    
    if ! command -v pip &> /dev/null; then
        missing_managers+=("pip")
    fi
    
    if ! command -v conda &> /dev/null; then
        missing_managers+=("conda")
    fi
    
    if [ ${#missing_managers[@]} -ne 0 ]; then
        echo "警告：以下包管理器未安装："
        for manager in "${missing_managers[@]}"; do
            echo "- $manager"
        done
        echo "部分功能可能无法使用。"
        echo "----------------------------------------"
    fi
}

# 定义各个镜像源的 URL
declare -A PIP_SOURCES=(
    ["1"]="中国科学技术大学 (USTC) https://mirrors.ustc.edu.cn/pypi/web/simple"
    ["2"]="清华大学 (Tsinghua) https://pypi.tuna.tsinghua.edu.cn/simple"
    ["3"]="豆瓣 (Douban) https://pypi.douban.com/simple"
    ["4"]="阿里云 (Alibaba) https://mirrors.aliyun.com/pypi/simple/"
    ["5"]="官方源 https://pypi.org/simple"
)

declare -A CONDA_SOURCES=(
    ["1"]="清华大学 (Tsinghua) https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/"
    ["2"]="中国科学技术大学 (USTC) https://mirrors.ustc.edu.cn/anaconda/pkgs/main/"
    ["3"]="阿里云 (Alibaba) https://mirrors.aliyun.com/anaconda/pkgs/main/"
    ["4"]="中科大 (USTC) https://mirrors.ustc.edu.cn/anaconda/pkgs/main/"
    ["5"]="官方源 https://repo.anaconda.com/pkgs/main/"
)

# 备份 conda 配置文件
backup_condarc() {
    if [ -f ~/.condarc ]; then
        cp ~/.condarc ~/.condarc.bak
        echo "已备份现有的 .condarc 至 .condarc.bak"
    fi
}

# 显示主菜单
show_main_menu() {
    echo "========================================"
    echo "      Python 包管理器源切换工具"
    echo "========================================"
    echo "请选择要配置的包管理器："
    echo "1) pip"
    echo "2) conda"
    echo "0) 退出"
    echo "========================================"
}

# 显示 pip 源菜单
show_pip_menu() {
    echo "========================================"
    echo "        pip 源切换菜单"
    echo "========================================"
    echo "请选择要设置的 pip 源："
    for key in "${!PIP_SOURCES[@]}"; do
        echo "$key) ${PIP_SOURCES[$key]}"
    done
    echo "9) 返回上级菜单"
    echo "0) 退出"
    echo "========================================"
}

# 显示 conda 源菜单
show_conda_menu() {
    echo "========================================"
    echo "        conda 源切换菜单"
    echo "========================================"
    echo "请选择要设置的 conda 源："
    for key in "${!CONDA_SOURCES[@]}"; do
        echo "$key) ${CONDA_SOURCES[$key]}"
    done
    echo "9) 返回上级菜单"
    echo "0) 退出"
    echo "========================================"
}

# 设置 pip 源
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
            url="https://pypi.douban.com/simple"
            ;;
        4)
            url="https://mirrors.aliyun.com/pypi/simple/"
            ;;
        5)
            url="https://pypi.org/simple"
            ;;
        9)
            return 1
            ;;
        0)
            echo "感谢使用！"
            exit 0
            ;;
        *)
            echo "无效的选择，请重新选择。"
            return 2
            ;;
    esac

    if [ $? -eq 0 ] && [ -n "$url" ]; then
        # 设置 pip 源
        pip config set global.index-url "$url"

        echo "----------------------------------------"
        echo "已将 pip 源设置为：$url"
        echo "当前 pip 源配置如下："
        pip config get global.index-url
        echo "----------------------------------------"
    fi
}

# 设置 conda 源
set_conda_source() {
    local choice=$1
    local url
    local name

    case $choice in
        1)
            url="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/"
            name="清华大学 (Tsinghua)"
            ;;
        2)
            url="https://mirrors.ustc.edu.cn/anaconda/pkgs/main/"
            name="中国科学技术大学 (USTC)"
            ;;
        3)
            url="https://mirrors.aliyun.com/anaconda/pkgs/main/"
            name="阿里云 (Alibaba)"
            ;;
        4)
            url="https://mirrors.ustc.edu.cn/anaconda/pkgs/main/"
            name="中科大 (USTC)"
            ;;
        5)
            url="https://repo.anaconda.com/pkgs/main/"
            name="官方源"
            ;;
        9)
            return 1
            ;;
        0)
            echo "感谢使用！"
            exit 0
            ;;
        *)
            echo "无效的选择，请重新选择。"
            return 2
            ;;
    esac

    if [ $? -eq 0 ] && [ -n "$url" ]; then
        # 备份现有配置
        backup_condarc

        # 清除所有现有的频道
        conda config --remove-key channels

        # 添加新的频道
        conda config --add channels "$url"

        # 设置默认频道
        conda config --set show_channel_urls yes

        echo "----------------------------------------"
        echo "已将 Conda 源设置为：$name - $url"
        echo "当前 Conda 源配置如下："
        cat ~/.condarc
        echo "----------------------------------------"
    fi
}

# 主函数
main() {
    # 检查包管理器安装状态
    check_package_managers

    while true; do
        show_main_menu
        read -p "请输入您的选择 [0-2]: " main_choice

        case $main_choice in
            1)  # pip 配置
                while true; do
                    show_pip_menu
                    read -p "请输入您的选择 [0-9]: " pip_choice
                    set_pip_source "$pip_choice"
                    if [ $? -eq 1 ]; then
                        break
                    fi
                done
                ;;
            2)  # conda 配置
                while true; do
                    show_conda_menu
                    read -p "请输入您的选择 [0-9]: " conda_choice
                    set_conda_source "$conda_choice"
                    if [ $? -eq 1 ]; then
                        break
                    fi
                done
                ;;
            0)  # 退出程序
                echo "感谢使用！"
                exit 0
                ;;
            *)  # 无效选择
                echo "无效的选择，请重新选择。"
                ;;
        esac
    done
}

# 运行主函数
main
