#!/bin/bash

# 定义支持的视频文件扩展名
VIDEO_EXTENSIONS=("mp4" "avi" "mkv" "mov" "flv" "wmv" "webm")

# 设置日志颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 函数：日志输出
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 函数：检查文件是否为视频文件
is_video_file() {
    local file="$1"
    local extension="${file##*.}"
    extension="${extension,,}" # 转为小写
    
    for ext in "${VIDEO_EXTENSIONS[@]}"; do
        if [[ "$extension" == "$ext" ]]; then
            return 0 # 是视频文件
        fi
    done
    return 1 # 不是视频文件
}

# 函数：处理视频，提取帧
process_video() {
    local VIDEO_PATH="$1"
    local FPS="${2:-1}" # 默认每秒提取1帧，可选参数

    # 检查视频文件是否存在
    if [ ! -f "$VIDEO_PATH" ]; then
        log_error "文件 '$VIDEO_PATH' 不存在。"
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
    
    # 确保目录权限正确
    chmod 755 "$OUTPUT_DIR"

    # 检查输出目录是否创建成功
    if [ ! -d "$OUTPUT_DIR" ]; then
        log_error "无法创建输出目录 '$OUTPUT_DIR'"
        return 1
    fi

    log_info "正在从 '$VIDEO_PATH' 提取帧..."
    local SUCCESS=false

    # 尝试不同方法提取帧
    log_info "尝试方法1 (基本方法)..."
    if ffmpeg -hide_banner -i "$VIDEO_PATH" -vf "fps=$FPS" "$OUTPUT_DIR/${VIDEO_NAME}_%04d.png" 2>/dev/null; then
        log_success "方法1成功: 帧已保存到目录: $OUTPUT_DIR"
        SUCCESS=true
    else
        log_warn "方法1失败，尝试其他方法..."
    fi

    if [ "$SUCCESS" = false ]; then
        log_info "尝试方法2 (颜色空间转换)..."
        if ffmpeg -hide_banner -i "$VIDEO_PATH" -vf "fps=$FPS,format=yuv420p" -pix_fmt rgb24 "$OUTPUT_DIR/${VIDEO_NAME}_%04d.png" 2>/dev/null; then
            log_success "方法2成功: 帧已保存到目录: $OUTPUT_DIR"
            SUCCESS=true
        else
            log_warn "方法2失败，尝试其他方法..."
        fi
    fi
    
    if [ "$SUCCESS" = false ]; then
        log_info "尝试方法3 (使用JPEG格式)..."
        if ffmpeg -hide_banner -i "$VIDEO_PATH" -vf "fps=$FPS" -q:v 2 "$OUTPUT_DIR/${VIDEO_NAME}_%04d.jpg" 2>/dev/null; then
            log_success "方法3成功: 帧已以JPEG格式保存到目录: $OUTPUT_DIR"
            SUCCESS=true
        else
            log_warn "方法3失败，尝试其他方法..."
        fi
    fi
    
    if [ "$SUCCESS" = false ]; then
        log_info "尝试方法4 (指定解码器)..."
        if ffmpeg -hide_banner -c:v h264 -i "$VIDEO_PATH" -vf "fps=$FPS" -q:v 2 "$OUTPUT_DIR/${VIDEO_NAME}_%04d.jpg" 2>/dev/null; then
            log_success "方法4成功: 使用h264解码器，帧已保存到目录: $OUTPUT_DIR"
            SUCCESS=true
        else
            log_warn "方法4失败，尝试其他方法..."
        fi
    fi
    
    if [ "$SUCCESS" = false ]; then
        log_info "尝试最终方法并显示详细错误..."
        local RESULT
        RESULT=$(ffmpeg -v verbose -i "$VIDEO_PATH" -vf "fps=$FPS" -q:v 3 "$OUTPUT_DIR/${VIDEO_NAME}_%04d.jpg" 2>&1)
        if [ $? -eq 0 ]; then
            log_success "最终方法成功: 帧已保存到目录: $OUTPUT_DIR"
            SUCCESS=true
        else
            log_error "所有方法均失败。视频文件可能已损坏或格式不受支持。"
            log_error "详细错误信息:"
            echo "$RESULT" | grep -i "error"
            return 1
        fi
    fi
    
    return 0
}

# 函数：将视频转换为 MP4 格式
convert_to_mp4() {
    local VIDEO_PATH="$1"

    # 检查视频文件是否存在
    if [ ! -f "$VIDEO_PATH" ]; then
        log_error "文件 '$VIDEO_PATH' 不存在。"
        return 1
    fi

    # 获取视频的目录和文件名（不含扩展名）
    local VIDEO_DIR
    VIDEO_DIR=$(dirname "$VIDEO_PATH")
    local VIDEO_FILENAME
    VIDEO_FILENAME=$(basename "$VIDEO_PATH")
    local VIDEO_NAME="${VIDEO_FILENAME%.*}"
    local VIDEO_EXTENSION="${VIDEO_FILENAME##*.}"
    VIDEO_EXTENSION="${VIDEO_EXTENSION,,}"

    # 如果已经是 mp4 格式，跳过转换
    if [[ "$VIDEO_EXTENSION" == "mp4" ]]; then
        log_info "文件 '$VIDEO_PATH' 已经是 MP4 格式，跳过转换。"
        return 0
    fi

    # 定义输出 MP4 文件路径
    local OUTPUT_PATH="$VIDEO_DIR/${VIDEO_NAME}.mp4"

    # 使用 ffmpeg 转换为 MP4 格式
    log_info "正在将 '$VIDEO_PATH' 转换为 MP4..."
    if ffmpeg -hide_banner -i "$VIDEO_PATH" -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k "$OUTPUT_PATH" >/dev/null 2>&1; then
        log_success "已将 '$VIDEO_PATH' 转换为 '$OUTPUT_PATH'"
        return 0
    else
        log_error "转换 '$VIDEO_PATH' 失败。"
        return 1
    fi
}

# 函数：将多个视频文件合并为一个 MP4 视频
merge_videos_in_directory() {
    local DIRECTORY="$1"
    local MERGED_VIDEO_PATH="$DIRECTORY/merged_output.mp4"

    # 确保目录存在
    if [ ! -d "$DIRECTORY" ]; then
        log_error "目录 '$DIRECTORY' 不存在"
        return 1
    fi

    # 查找目录中的视频文件，并按名称排序
    local VIDEO_FILES=()
    while IFS= read -r -d '' file; do
        if is_video_file "$file"; then
            VIDEO_FILES+=("$file")
        fi
    done < <(find "$DIRECTORY" -maxdepth 1 -type f -print0 | sort -z)

    local VIDEO_COUNT=${#VIDEO_FILES[@]}

    if [ "$VIDEO_COUNT" -lt 2 ]; then
        log_warn "在目录 '$DIRECTORY' 中找到的视频文件少于2个，跳过合并。"
        return 0
    fi

    log_info "在目录 '$DIRECTORY' 中找到 $VIDEO_COUNT 个视频文件用于合并"

    # 创建临时文件列表
    local TEMP_FILE_LIST="$DIRECTORY/file_list.txt"
    > "$TEMP_FILE_LIST"
    
    for video in "${VIDEO_FILES[@]}"; do
        # 使用绝对路径并正确处理特殊字符
        echo "file '$(realpath "$video")'" >> "$TEMP_FILE_LIST"
    done

    # 使用 ffmpeg 的 concat demuxer 进行合并
    log_info "正在合并视频文件..."
    if ffmpeg -hide_banner -f concat -safe 0 -i "$TEMP_FILE_LIST" -c copy "$MERGED_VIDEO_PATH" >/dev/null 2>&1; then
        log_success "已合并目录 '$DIRECTORY' 中的 $VIDEO_COUNT 个视频为 '$MERGED_VIDEO_PATH'"
    else
        log_error "合并目录 '$DIRECTORY' 中的视频失败。"
    fi

    # 删除临时文件列表
    rm -f "$TEMP_FILE_LIST"
}

# 显示使用指南
show_usage() {
    echo "=============================================="
    echo "          视频处理脚本使用指南"
    echo "=============================================="
    echo "用法: $0 <command> /path/to/video_or_directory [options]"
    echo ""
    echo "命令说明:"
    echo "  getframe      提取视频帧"
    echo "  getmp4        将非 MP4 视频转换为 MP4 格式"
    echo "  mergevideo    合并目录中的视频文件（递归处理每个子目录）"
    echo ""
    echo "参数说明:"
    echo "  /path/to/video_or_directory  指定单个视频文件或包含视频文件的目录。"
    echo ""
    echo "选项:"
    echo "  -f, --fps <帧率>      提取帧时的帧率（默认为1）"
    echo "  -h, --help            显示此帮助信息"
    echo ""
    echo "支持的视频格式:"
    echo "  mp4, avi, mkv, mov, flv, wmv, webm"
    echo "=============================================="
}

# 处理输入目录或文件（提取帧）
process_getframe() {
    local path="$1"
    local fps="${2:-1}"  # 默认为1
    
    if [ ! -e "$path" ]; then
        log_error "路径 '$path' 不存在。"
        return 1
    fi

    if [ -d "$path" ]; then
        log_info "检测到目录: $path"
        log_info "开始递归处理目录及其子目录中的视频文件..."
        local video_files=()
        local count=0

        # 查找所有视频文件
        while IFS= read -r -d '' file; do
            if is_video_file "$file"; then
                video_files+=("$file")
                ((count++))
            fi
        done < <(find "$path" -type f -print0)

        log_info "在目录 '$path' 及其子目录中找到 $count 个视频文件"
        
        if [ $count -eq 0 ]; then
            log_warn "没有找到视频文件，退出处理。"
            return 0
        fi

        # 处理每个视频文件
        for file in "${video_files[@]}"; do
            log_info "处理视频文件: $file"
            process_video "$file" "$fps"
        done
        
        log_success "所有视频文件处理完成。"
    elif [ -f "$path" ]; then
        if is_video_file "$path"; then
            log_info "处理单个视频文件: $path"
            process_video "$path" "$fps"
            log_success "视频文件处理完成。"
        else
            log_error "文件 '$path' 不是支持的视频格式。"
            return 1
        fi
    else
        log_error "'$path' 既不是文件也不是目录。"
        return 1
    fi
}

# 处理输入目录或文件（转换为MP4）
process_getmp4() {
    local path="$1"
    
    if [ ! -e "$path" ]; then
        log_error "路径 '$path' 不存在。"
        return 1
    fi
    
    if [ -d "$path" ]; then
        log_info "检测到目录: $path"
        log_info "开始转换目录中的非 MP4 视频文件..."
        local video_files=()
        local count=0

        # 查找所有非MP4视频文件
        while IFS= read -r -d '' file; do
            if is_video_file "$file" && [[ "${file,,}" != *".mp4" ]]; then
                video_files+=("$file")
                ((count++))
            fi
        done < <(find "$path" -type f -print0)

        log_info "在目录 '$path' 及其子目录中找到 $count 个非MP4视频文件"
        
        if [ $count -eq 0 ]; then
            log_warn "没有找到非MP4视频文件，退出处理。"
            return 0
        fi

        # 处理每个视频文件
        for file in "${video_files[@]}"; do
            log_info "转换视频文件: $file"
            convert_to_mp4 "$file"
        done
        
        log_success "所有视频文件转换完成。"
    elif [ -f "$path" ]; then
        if is_video_file "$path"; then
            convert_to_mp4 "$path"
        else
            log_error "文件 '$path' 不是支持的视频格式。"
            return 1
        fi
    else
        log_error "'$path' 既不是文件也不是目录。"
        return 1
    fi
}

# 处理输入目录（合并视频）
process_mergevideo() {
    local path="$1"
    
    if [ ! -e "$path" ]; then
        log_error "路径 '$path' 不存在。"
        return 1
    fi
    
    if [ -d "$path" ]; then
        log_info "检测到目录: $path"
        log_info "开始递归合并目录中的视频文件..."
        
        # 获取所有子目录
        local directories=()
        while IFS= read -r -d '' dir; do
            directories+=("$dir")
        done < <(find "$path" -type d -print0)
        
        local dir_count=${#directories[@]}
        log_info "在 '$path' 中找到 $dir_count 个目录"
        
        # 遍历每个目录，合并视频
        for dir in "${directories[@]}"; do
            log_info "处理目录: $dir"
            merge_videos_in_directory "$dir"
        done
        
        log_success "所有目录中的视频文件合并完成。"
    elif [ -f "$path" ]; then
        log_error "'mergevideo' 命令需要一个目录作为输入。"
        return 1
    else
        log_error "'$path' 既不是文件也不是目录。"
        return 1
    fi
}

# 主程序
main() {
    # 检查参数数量
    if [ $# -lt 1 ]; then
        show_usage
        exit 1
    fi
    
    # 解析命令
    local COMMAND="$1"
    shift
    
    # 解析选项
    local INPUT_PATH=""
    local FPS=1
    
    while [ $# -gt 0 ]; do
        case "$1" in
            -f|--fps)
                FPS="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -*)
                log_error "未知选项: $1"
                show_usage
                exit 1
                ;;
            *)
                if [ -z "$INPUT_PATH" ]; then
                    INPUT_PATH="$1"
                else
                    log_error "多余的参数: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # 检查必需参数
    if [ -z "$INPUT_PATH" ]; then
        log_error "缺少路径参数"
        show_usage
        exit 1
    fi
    
    # 检查必需的命令
    if ! command -v ffmpeg >/dev/null 2>&1; then
        log_error "未找到 ffmpeg 命令。请安装 ffmpeg。"
        exit 1
    fi
    
    # 执行相应的命令
    case "$COMMAND" in
        getframe)
            process_getframe "$INPUT_PATH" "$FPS"
            ;;
        getmp4)
            process_getmp4 "$INPUT_PATH"
            ;;
        mergevideo)
            process_mergevideo "$INPUT_PATH"
            ;;
        *)
            log_error "未知命令: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
}

# 执行主程序
main "$@"
