#!/bin/bash

# 定义支持的视频文件扩展名
VIDEO_EXTENSIONS=("mp4" "avi" "mkv" "mov" "flv" "wmv" "webm")

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

# 函数：处理单个视频文件，提取帧
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

# 函数：将视频转换为 MP4 格式
convert_to_mp4() {
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
    local VIDEO_EXTENSION="${VIDEO_FILENAME##*.}"

    # 如果已经是 mp4 格式，跳过转换
    if [[ "${VIDEO_EXTENSION,,}" == "mp4" ]]; then
        echo "文件 '$VIDEO_PATH' 已经是 MP4 格式，跳过转换。"
        return 0
    fi

    # 定义输出 MP4 文件路径
    local OUTPUT_PATH="$VIDEO_DIR/${VIDEO_NAME}.mp4"

    # 使用 ffmpeg 转换为 MP4 格式
    ffmpeg -i "$VIDEO_PATH" -c:v libx264 -c:a aac "$OUTPUT_PATH" >/dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "已将 '$VIDEO_PATH' 转换为 '$OUTPUT_PATH'"
    else
        echo "转换 '$VIDEO_PATH' 失败。"
    fi
}

# 函数：将多个视频文件合并为一个 MP4 视频
merge_videos_in_directory() {
    local DIRECTORY="$1"
    local MERGED_VIDEO_PATH="$DIRECTORY/merged_output.mp4"

    # 查找目录中的视频文件，并按名称排序
    mapfile -t VIDEO_FILES < <(find "$DIRECTORY" -maxdepth 1 -type f \( $(printf -- "-iname *.%s -o " "${VIDEO_EXTENSIONS[@]}") -false \) | sort)

    # 过滤出支持的视频文件
    VIDEO_FILES=($(printf "%s\n" "${VIDEO_FILES[@]}" | grep -Ei "\.($(printf '|%s' "${VIDEO_EXTENSIONS[@]}"))$"))

    local VIDEO_COUNT=${#VIDEO_FILES[@]}

    if [ "$VIDEO_COUNT" -lt 2 ]; then
        echo "在目录 '$DIRECTORY' 中的视频文件少于2个，跳过合并。"
        return
    fi

    # 创建临时文件列表
    local TEMP_FILE_LIST="$DIRECTORY/file_list.txt"
    > "$TEMP_FILE_LIST"
    for video in "${VIDEO_FILES[@]}"; do
        # 使用绝对路径并处理空格
        echo "file '$(realpath "$video")'" >> "$TEMP_FILE_LIST"
    done

    # 使用 ffmpeg 的 concat demuxer 进行合并
    ffmpeg -f concat -safe 0 -i "$TEMP_FILE_LIST" -c copy "$MERGED_VIDEO_PATH" >/dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "已合并目录 '$DIRECTORY' 中的 $VIDEO_COUNT 个视频为 '$MERGED_VIDEO_PATH'"
    else
        echo "合并目录 '$DIRECTORY' 中的视频失败。"
    fi

    # 删除临时文件列表
    rm -f "$TEMP_FILE_LIST"
}

# 显示使用指南
show_usage() {
    echo "=============================================="
    echo "          视频处理脚本使用指南"
    echo "=============================================="
    echo "用法: $0 <command> /path/to/video_or_directory"
    echo ""
    echo "命令说明:"
    echo "  getframe      提取视频帧"
    echo "  getmp4        将非 MP4 视频转换为 MP4 格式"
    echo "  mergevideo    合并目录中的视频文件（递归处理每个子目录）"
    echo ""
    echo "参数说明:"
    echo "  /path/to/video_or_directory  指定单个视频文件或包含视频文件的目录。"
    echo ""
    echo "支持的视频格式:"
    echo "  mp4, avi, mkv, mov, flv, wmv, webm"
    echo "=============================================="
}

# 检查是否提供了至少两个参数
if [ $# -lt 2 ]; then
    show_usage
    exit 1
fi

COMMAND="$1"
INPUT_PATH="$2"

# 检查输入路径是否存在
if [ ! -e "$INPUT_PATH" ]; then
    echo "错误: 路径 '$INPUT_PATH' 不存在。"
    exit 1
fi

# 根据命令执行相应的功能
case "$COMMAND" in
    getframe)
        # 提取视频帧功能
        process_input() {
            local path="$1"
            if [ -d "$path" ]; then
                echo "检测到目录: $path"
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
                done < <(find "$path" -type f \( "${FIND_CONDITIONS[@]}" \) -print0)
                echo "所有视频文件处理完成。"
            elif [ -f "$path" ]; then
                if is_video_file "$path"; then
                    echo "处理单个视频文件: $path"
                    process_video "$path"
                    echo "视频文件处理完成。"
                else
                    echo "错误: 文件 '$path' 不是支持的视频格式。"
                    exit 1
                fi
            else
                echo "错误: '$path' 既不是文件也不是目录。"
                exit 1
            fi
        }

        process_input "$INPUT_PATH"
        ;;
    getmp4)
        # 转换视频为 MP4 格式功能
        convert_input() {
            local path="$1"
            if [ -d "$path" ]; then
                echo "检测到目录: $path"
                echo "开始转换目录中的非 MP4 视频文件..."

                # 构造 find 命令的搜索条件，排除 mp4
                FIND_CONDITIONS=()
                for ext in "${VIDEO_EXTENSIONS[@]}"; do
                    if [[ "${ext,,}" != "mp4" ]]; then
                        FIND_CONDITIONS+=("-iname" "*.${ext}")
                        FIND_CONDITIONS+=("-o")
                    fi
                done
                # 去掉最后的 -o
                unset 'FIND_CONDITIONS[-1]'

                # 执行 find 命令并读取结果
                while IFS= read -r -d '' file; do
                    if is_video_file "$file"; then
                        echo "转换视频文件: $file"
                        convert_to_mp4 "$file"
                    else
                        echo "跳过非视频文件: $file"
                    fi
                done < <(find "$path" -type f \( "${FIND_CONDITIONS[@]}" \) -print0)
                echo "所有视频文件转换完成。"
            elif [ -f "$path" ]; then
                if is_video_file "$path"; then
                    convert_to_mp4 "$path"
                else
                    echo "错误: 文件 '$path' 不是支持的视频格式。"
                    exit 1
                fi
            else
                echo "错误: '$path' 既不是文件也不是目录。"
                exit 1
            fi
        }

        convert_input "$INPUT_PATH"
        ;;
    mergevideo)
        # 合并视频文件功能
        mergevideo_input() {
            local path="$1"
            if [ -d "$path" ]; then
                echo "检测到目录: $path"
                echo "开始递归合并目录中的视频文件..."

                # 使用 find 递归遍历所有子目录
                find "$path" -type d | while IFS= read -r dir; do
                    echo "处理目录: $dir"
                    merge_videos_in_directory "$dir"
                done
                echo "所有目录中的视频文件合并完成。"
            elif [ -f "$path" ]; then
                echo "错误: 'mergevideo' 命令需要一个目录作为输入。"
                exit 1
            else
                echo "错误: '$path' 既不是文件也不是目录。"
                exit 1
            fi
        }

        mergevideo_input "$INPUT_PATH"
        ;;
    *)
        echo "错误: 未知命令 '$COMMAND'。"
        show_usage
        exit 1
        ;;
esac
