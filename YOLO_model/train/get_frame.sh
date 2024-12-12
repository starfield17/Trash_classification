#!/bin/bash

# 检查是否提供了视频路径
if [ -z "$1" ]; then
    echo "用法: $0 /path/to/video"
    exit 1
fi

# 获取视频的完整路径
VIDEO_PATH="$1"

# 检查视频文件是否存在
if [ ! -f "$VIDEO_PATH" ]; then
    echo "错误: 文件 '$VIDEO_PATH' 不存在。"
    exit 1
fi

# 获取视频的目录和文件名（不含扩展名）
VIDEO_DIR=$(dirname "$VIDEO_PATH")
VIDEO_FILENAME=$(basename "$VIDEO_PATH")
VIDEO_NAME="${VIDEO_FILENAME%.*}"

# 创建保存图片的目录
OUTPUT_DIR="$VIDEO_DIR/${VIDEO_NAME}_frames"
mkdir -p "$OUTPUT_DIR"

# 使用ffmpeg每1秒抽取一帧
# 输出格式为 视频名_X.png，其中X从1开始递增
ffmpeg -i "$VIDEO_PATH" -vf fps=1 "$OUTPUT_DIR/${VIDEO_NAME}_%d.png"

echo "帧已保存到目录: $OUTPUT_DIR"
