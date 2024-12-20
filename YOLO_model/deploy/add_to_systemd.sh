#!/bin/bash

# add_to_systemd.sh
# 用法:
#   安装服务: bash add_to_systemd.sh install "conda_env_name" "/path/to/script.py"
#   移除服务: bash add_to_systemd.sh remove "/path/to/script.py"

echo "==============================="
echo "初始化启动脚本: add_to_systemd.sh"
echo "==============================="

# 检查是否至少有一个参数
if [ "$#" -lt 1 ]; then
    echo "错误: 参数数量不正确。"
    echo "用法:"
    echo "  安装服务: $0 install \"conda_env_name\" \"/path/to/script.py\""
    echo "  移除服务: $0 remove \"/path/to/script.py\""
    exit 1
fi

ACTION=$1

# 函数：安装服务
install_service() {
    if [ "$#" -ne 3 ]; then
        echo "错误: 安装服务需要三个参数。"
        echo "用法: $0 install \"conda_env_name\" \"/path/to/script.py\""
        exit 1
    fi

    CONDA_ENV=$2
    SCRIPT_PATH=$3

    echo "Conda 环境: ${CONDA_ENV:-'系统默认 Python 环境'}"
    echo "Python 脚本路径: $SCRIPT_PATH"

    # 检查脚本是否存在
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "错误: 脚本文件 '$SCRIPT_PATH' 不存在。"
        exit 1
    fi

    # 获取当前用户名
    USER_NAME=$(whoami)
    USER_HOME=$(eval echo "~$USER_NAME")

    echo "当前用户: $USER_NAME"
    echo "用户主目录: $USER_HOME"

    # 定义服务名称
    SERVICE_NAME=$(basename "$SCRIPT_PATH" .py).service

    echo "将创建的 systemd 服务名称: $SERVICE_NAME"

    # 检查是否已有同名服务存在
    if systemctl list-unit-files | grep -q "^$SERVICE_NAME"; then
        echo "警告: 服务 '$SERVICE_NAME' 已存在。准备覆盖该服务。"
    fi

    # 创建 systemd 服务文件内容
    SERVICE_FILE="[Unit]
Description=Auto-start $SERVICE_NAME
After=network.target

[Service]
Type=simple
User=$USER_NAME
Environment=PATH=/usr/bin:/bin:/usr/local/bin
WorkingDirectory=$(dirname "$SCRIPT_PATH")
ExecStart=/bin/bash -c '
if [ -n \"$CONDA_ENV\" ]; then
    echo \"激活 Conda 环境: $CONDA_ENV\"
    # 查找conda的安装路径
    CONDA_BASE=\$(conda info --base 2>/dev/null)
    if [ -z \"\$CONDA_BASE\" ]; then
        echo \"错误: Conda 未安装或未在 PATH 中找到。\"
        exit 1
    fi
    source \"\$CONDA_BASE/etc/profile.d/conda.sh\"
    conda activate \"$CONDA_ENV\"
    if [ \"\$?\" -ne 0 ]; then
        echo \"错误: 无法激活 Conda 环境 '$CONDA_ENV'。\"
        exit 1
    fi
else
    echo \"使用系统默认的 Python 环境。\"
fi

echo \"开始执行 Python 脚本: $SCRIPT_PATH\"
python \"$SCRIPT_PATH\"
'

Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target"

    echo "正在创建 systemd 服务文件..."

    # 将服务文件写入 /etc/systemd/system/
    echo "$SERVICE_FILE" | sudo tee /etc/systemd/system/$SERVICE_NAME > /dev/null

    if [ "$?" -ne 0 ]; then
        echo "错误: 无法写入服务文件到 /etc/systemd/system/$SERVICE_NAME。请检查权限。"
        exit 1
    fi

    echo "服务文件已创建: /etc/systemd/system/$SERVICE_NAME"

    # 重新加载 systemd 守护进程
    echo "正在重新加载 systemd 守护进程..."
    sudo systemctl daemon-reload

    if [ "$?" -ne 0 ]; then
        echo "错误: 重新加载 systemd 守护进程失败。"
        exit 1
    fi

    # 启用服务，使其在系统启动时自动运行
    echo "正在启用服务 '$SERVICE_NAME'..."
    sudo systemctl enable $SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "错误: 启用服务 '$SERVICE_NAME' 失败。"
        exit 1
    fi

    # 启动服务
    echo "正在启动服务 '$SERVICE_NAME'..."
    sudo systemctl start $SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "错误: 启动服务 '$SERVICE_NAME' 失败。"
        exit 1
    fi

    echo "=========================================="
    echo "服务 '$SERVICE_NAME' 已成功创建并启动。"
    echo "它将在系统启动时自动运行。"
    echo "=========================================="

    echo "你可以使用以下命令来检查服务状态:"
    echo "  systemctl status $SERVICE_NAME"
    echo "查看服务日志使用:"
    echo "  journalctl -u $SERVICE_NAME -f"
}

# 函数：移除服务
remove_service() {
    if [ "$#" -ne 2 ]; then
        echo "错误: 移除服务需要两个参数。"
        echo "用法: $0 remove \"/path/to/script.py\""
        exit 1
    fi

    SCRIPT_PATH=$2

    echo "Python 脚本路径: $SCRIPT_PATH"

    # 检查脚本是否存在
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "错误: 脚本文件 '$SCRIPT_PATH' 不存在。"
        # 继续尝试移除服务文件，即使脚本不存在
    fi

    # 定义服务名称
    SERVICE_NAME=$(basename "$SCRIPT_PATH" .py).service

    echo "将移除的 systemd 服务名称: $SERVICE_NAME"

    # 检查服务是否存在
    if ! systemctl list-unit-files | grep -q "^$SERVICE_NAME"; then
        echo "错误: 服务 '$SERVICE_NAME' 不存在。"
        exit 1
    fi

    # 停止服务
    echo "正在停止服务 '$SERVICE_NAME'..."
    sudo systemctl stop $SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "警告: 停止服务 '$SERVICE_NAME' 失败。可能服务已停止。"
    else
        echo "服务 '$SERVICE_NAME' 已停止。"
    fi

    # 禁用服务
    echo "正在禁用服务 '$SERVICE_NAME'..."
    sudo systemctl disable $SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "警告: 禁用服务 '$SERVICE_NAME' 失败。"
    else
        echo "服务 '$SERVICE_NAME' 已禁用。"
    fi

    # 删除服务文件
    echo "正在删除服务文件 '/etc/systemd/system/$SERVICE_NAME'..."
    sudo rm -f /etc/systemd/system/$SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "错误: 无法删除服务文件 '/etc/systemd/system/$SERVICE_NAME'。"
        exit 1
    fi

    # 重新加载 systemd 守护进程
    echo "正在重新加载 systemd 守护进程..."
    sudo systemctl daemon-reload

    if [ "$?" -ne 0 ]; then
        echo "错误: 重新加载 systemd 守护进程失败。"
        exit 1
    fi

    echo "=========================================="
    echo "服务 '$SERVICE_NAME' 已成功移除。"
    echo "=========================================="

    echo "你可以使用以下命令来验证服务已被移除:"
    echo "  systemctl status $SERVICE_NAME"
}

# 根据 ACTION 调用相应的函数
case "$ACTION" in
    install)
        install_service "$@"
        ;;
    remove)
        remove_service "$@"
        ;;
    *)
        echo "错误: 无效的操作 '$ACTION'。"
        echo "支持的操作: install, remove"
        exit 1
        ;;
esac
