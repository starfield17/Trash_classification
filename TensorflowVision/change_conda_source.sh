#!/bin/bash

# 检查 conda 是否已安装
if ! command -v conda &> /dev/null
then
    echo "Conda 未安装。请先安装 Conda 后再运行此脚本。"
    exit 1
fi

# 备份现有的 .condarc 配置文件
backup_condarc() {
    if [ -f ~/.condarc ]; then
        cp ~/.condarc ~/.condarc.bak
        echo "已备份现有的 .condarc 至 .condarc.bak"
    fi
}

# 定义各个镜像源的 URL
declare -A CONDA_SOURCES=(
    ["1"]="清华大学 (Tsinghua) https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/"
    ["2"]="中国科学技术大学 (USTC) https://mirrors.ustc.edu.cn/anaconda/pkgs/main/"
    ["3"]="阿里云 (Alibaba) https://mirrors.aliyun.com/anaconda/pkgs/main/"
    ["4"]="中科大 (USTC) https://mirrors.ustc.edu.cn/anaconda/pkgs/main/"
    ["5"]="官方源 https://repo.anaconda.com/pkgs/main/"
)

# 函数：显示菜单
show_menu() {
    echo "========================================"
    echo "            Conda 源切换脚本"
    echo "========================================"
    echo "请选择要设置的 Conda 源："
    for key in "${!CONDA_SOURCES[@]}"; do
        echo "$key) ${CONDA_SOURCES[$key]}"
    done
    echo "0) 退出"
    echo "========================================"
}

# 函数：设置 Conda 源
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
        0)
            echo "已退出设置。"
            exit 0
            ;;
        *)
            echo "无效的选择，请重新选择。"
            return
            ;;
    esac

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
}

# 主循环
while true; do
    show_menu
    read -p "请输入您的选择 [0-5]: " user_choice
    set_conda_source "$user_choice"
done
