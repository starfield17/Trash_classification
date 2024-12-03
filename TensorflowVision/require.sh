conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/r/
conda config --set show_channel_urls yes
declare -A PIP_SOURCES=(
    ["1"]="中国科学技术大学 (USTC) https://mirrors.ustc.edu.cn/pypi/web/simple"
    ["2"]="清华大学 (Tsinghua) https://pypi.tuna.tsinghua.edu.cn/simple"
    ["3"]="豆瓣 (Douban) https://pypi.douban.com/simple"
    ["4"]="阿里云 (Alibaba) https://mirrors.aliyun.com/pypi/simple/"
    ["5"]="官方源 https://pypi.org/simple"
)

# 函数：显示菜单
show_menu() {
    echo "========================================"
    echo "        pip 源切换脚本"
    echo "========================================"
    echo "请选择要设置的 pip 源："
    for key in "${!PIP_SOURCES[@]}"; do
        echo "$key) ${PIP_SOURCES[$key]}"
    done
    echo "0) 退出"
    echo "========================================"
}

# 函数：设置 pip 源
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
        0)
            echo "已退出设置。"
            exit 0
            ;;
        *)
            echo "无效的选择，请重新选择。"
            return
            ;;
    esac

    # 设置 pip 源
    pip config set global.index-url "$url"

    echo "----------------------------------------"
    echo "已将 pip 源设置为：$url"
    echo "当前 pip 源配置如下："
    pip config get global.index-url
    echo "----------------------------------------"
}

# 检查 pip 是否已安装
if ! command -v pip &> /dev/null
then
    echo "pip 未安装。请先安装 pip 后再运行此脚本。"
    exit 1
fi

# 主循环
while true; do
    show_menu
    read -p "请输入您的选择 [0-5]: " user_choice
    set_pip_source "$user_choice"
done

pip install tensorflow opencv-python numpy
#一眼丁真，鉴定为纯纯的trash
