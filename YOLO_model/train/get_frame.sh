#!/bin/bash

# 定义支持的视频文件扩展名
VIDEO_EXTENSIONS=("mp4" "avi" "mkv" "mov" "flv" "wmv")

# 函数：检查文件是否为视频文件
is_video_file() {
    local file="$1"
    local extension="${file##*.}"
    for ext in "${VIDEO_EXTENSIONS[@]}"; do
        if [[ "${extension,,}" == "$ext" ]]; then
            return 0
        fi
    done
    return 1
}

# 函数：处理单个视频文件
process_video() {
    local VIDEO_PATH="$1"

    # 检查视频文件是否存在
    if [ ! -f "$VIDEO_PATH" ]; then
        echo "错误: 文件 '$VIDEO_PATH' 不存在。"
        return 1
    fi

    # 获取视频的目录和文件名（不含扩展名）
    local VIDEO_DIR
    VIDEO_DIR=$(dirname "$VIDEO_PATH")
    local VIDEO_FILENAME
    VIDEO_FILENAME=$(basename "$VIDEO_PATH")
    local VIDEO_NAME="${VIDEO_FILENAME%.*}"

    # 创建保存帧的目录
    local OUTPUT_DIR="$VIDEO_DIR/${VIDEO_NAME}_frames"
    mkdir -p "$OUTPUT_DIR"

    # 使用 ffmpeg 每1秒抽取一帧，并指定输出格式和质量
    ffmpeg -i "$VIDEO_PATH" -vf fps=1 "$OUTPUT_DIR/${VIDEO_NAME}_%d.png" >/dev/null 2>&1

    echo "帧已保存到目录: $OUTPUT_DIR"
}

# 检查是否提供了路径
if [ -z "$1" ]; then
    echo "=============================================="
    echo "          视频帧提取脚本使用指南"
    echo "=============================================="
    echo "用法: $0 /path/to/video_or_directory"
    echo ""
    echo "参数说明:"
    echo "  /path/to/video_or_directory  指定单个视频文件或包含视频文件的目录。"
    echo ""
    echo "示例:"
    echo "  $0 /var/home/user/Videos/sample.mp4"
    echo "  $0 /var/home/user/Videos/"
    echo ""
    echo "支持的视频格式:"
    echo "  mp4, avi, mkv, mov, flv, wmv"
    echo "=============================================="
    exit 1
fi

INPUT_PATH="$1"

# 判断输入是文件还是目录
if [ -d "$INPUT_PATH" ]; then
    echo "检测到目录: $INPUT_PATH"
    echo "开始处理目录中的视频文件..."
    # 构造 find 命令的搜索条件
    FIND_CONDITIONS=()
    for ext in "${VIDEO_EXTENSIONS[@]}"; do
        FIND_CONDITIONS+=("-iname" "*.${ext}")
        FIND_CONDITIONS+=("-o")
    done
    # 去掉最后的 -o
    unset 'FIND_CONDITIONS[-1]'

    # 执行 find 命令并读取结果
    while IFS= read -r -d '' file; do
        if is_video_file "$file"; then
            echo "处理视频文件: $file"
            process_video "$file"
        else
            echo "跳过非视频文件: $file"
        fi
    done < <(find "$INPUT_PATH" -type f \( "${FIND_CONDITIONS[@]}" \) -print0)
    echo "所有视频文件处理完成。"
else
    if is_video_file "$INPUT_PATH"; then
        echo "处理单个视频文件: $INPUT_PATH"
        process_video "$INPUT_PATH"
        echo "视频文件处理完成。"
    else
        echo "错误: 文件 '$INPUT_PATH' 不是支持的视频格式。"
        exit 1
    fi
fi
