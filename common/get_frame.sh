#!/bin/bash

# Define supported video file extensions
VIDEO_EXTENSIONS=("mp4" "avi" "mkv" "mov" "flv" "wmv" "webm")

# Set log colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function: Log output
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

# Function: Check if a file is a video file
is_video_file() {
    local file="$1"
    local extension="${file##*.}"
    extension="${extension,,}" # Convert to lowercase
    
    for ext in "${VIDEO_EXTENSIONS[@]}"; do
        if [[ "$extension" == "$ext" ]]; then
            return 0 # Is a video file
        fi
    done
    return 1 # Not a video file
}

# Function: Process video, extract frames
process_video() {
    local VIDEO_PATH="$1"
    local FPS="${2:-1}" # Default to extracting 1 frame per second, optional parameter

    # Check if video file exists
    if [ ! -f "$VIDEO_PATH" ]; then
        log_error "File '$VIDEO_PATH' does not exist."
        return 1
    fi

    # Get video directory and filename (without extension)
    local VIDEO_DIR
    VIDEO_DIR=$(dirname "$VIDEO_PATH")
    local VIDEO_FILENAME
    VIDEO_FILENAME=$(basename "$VIDEO_PATH")
    local VIDEO_NAME="${VIDEO_FILENAME%.*}"

    # Create directory to save frames
    local OUTPUT_DIR="$VIDEO_DIR/${VIDEO_NAME}_frames"
    mkdir -p "$OUTPUT_DIR"
    
    # Ensure directory permissions are correct
    chmod 755 "$OUTPUT_DIR"

    # Check if output directory was created successfully
    if [ ! -d "$OUTPUT_DIR" ]; then
        log_error "Could not create output directory '$OUTPUT_DIR'"
        return 1
    fi

    log_info "Extracting frames from '$VIDEO_PATH'..."
    local SUCCESS=false

    # Try different methods to extract frames
    log_info "Attempting method 1 (basic method)..."
    if ffmpeg -hide_banner -i "$VIDEO_PATH" -vf "fps=$FPS" "$OUTPUT_DIR/${VIDEO_NAME}_%04d.png" 2>/dev/null; then
        log_success "Method 1 successful: Frames saved to directory: $OUTPUT_DIR"
        SUCCESS=true
    else
        log_warn "Method 1 failed, trying other methods..."
    fi

    if [ "$SUCCESS" = false ]; then
        log_info "Attempting method 2 (color space conversion)..."
        if ffmpeg -hide_banner -i "$VIDEO_PATH" -vf "fps=$FPS,format=yuv420p" -pix_fmt rgb24 "$OUTPUT_DIR/${VIDEO_NAME}_%04d.png" 2>/dev/null; then
            log_success "Method 2 successful: Frames saved to directory: $OUTPUT_DIR"
            SUCCESS=true
        else
            log_warn "Method 2 failed, trying other methods..."
        fi
    fi
    
    if [ "$SUCCESS" = false ]; then
        log_info "Attempting method 3 (using JPEG format)..."
        if ffmpeg -hide_banner -i "$VIDEO_PATH" -vf "fps=$FPS" -q:v 2 "$OUTPUT_DIR/${VIDEO_NAME}_%04d.jpg" 2>/dev/null; then
            log_success "Method 3 successful: Frames saved in JPEG format to directory: $OUTPUT_DIR"
            SUCCESS=true
        else
            log_warn "Method 3 failed, trying other methods..."
        fi
    fi
    
    if [ "$SUCCESS" = false ]; then
        log_info "Attempting method 4 (specifying decoder)..."
        if ffmpeg -hide_banner -c:v h264 -i "$VIDEO_PATH" -vf "fps=$FPS" -q:v 2 "$OUTPUT_DIR/${VIDEO_NAME}_%04d.jpg" 2>/dev/null; then
            log_success "Method 4 successful: Used h264 decoder, frames saved to directory: $OUTPUT_DIR"
            SUCCESS=true
        else
            log_warn "Method 4 failed, trying other methods..."
        fi
    fi
    
    if [ "$SUCCESS" = false ]; then
        log_info "Attempting final method and showing detailed errors..."
        local RESULT
        RESULT=$(ffmpeg -v verbose -i "$VIDEO_PATH" -vf "fps=$FPS" -q:v 3 "$OUTPUT_DIR/${VIDEO_NAME}_%04d.jpg" 2>&1)
        if [ $? -eq 0 ]; then
            log_success "Final method successful: Frames saved to directory: $OUTPUT_DIR"
            SUCCESS=true
        else
            log_error "All methods failed. Video file may be corrupted or format not supported."
            log_error "Detailed error information:"
            echo "$RESULT" | grep -i "error"
            return 1
        fi
    fi
    
    return 0
}

# Function: Convert video to MP4 format
convert_to_mp4() {
    local VIDEO_PATH="$1"

    # Check if video file exists
    if [ ! -f "$VIDEO_PATH" ]; then
        log_error "File '$VIDEO_PATH' does not exist."
        return 1
    fi

    # Get video directory and filename (without extension)
    local VIDEO_DIR
    VIDEO_DIR=$(dirname "$VIDEO_PATH")
    local VIDEO_FILENAME
    VIDEO_FILENAME=$(basename "$VIDEO_PATH")
    local VIDEO_NAME="${VIDEO_FILENAME%.*}"
    local VIDEO_EXTENSION="${VIDEO_FILENAME##*.}"
    VIDEO_EXTENSION="${VIDEO_EXTENSION,,}"

    # If already in mp4 format, skip conversion
    if [[ "$VIDEO_EXTENSION" == "mp4" ]]; then
        log_info "File '$VIDEO_PATH' is already in MP4 format, skipping conversion."
        return 0
    fi

    # Define output MP4 file path
    local OUTPUT_PATH="$VIDEO_DIR/${VIDEO_NAME}.mp4"

    # Use ffmpeg to convert to MP4 format
    log_info "Converting '$VIDEO_PATH' to MP4..."
    if ffmpeg -hide_banner -i "$VIDEO_PATH" -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k "$OUTPUT_PATH" >/dev/null 2>&1; then
        log_success "Converted '$VIDEO_PATH' to '$OUTPUT_PATH'"
        return 0
    else
        log_error "Conversion of '$VIDEO_PATH' failed."
        return 1
    fi
}

# Function: Merge multiple video files into a single MP4 video
merge_videos_in_directory() {
    local DIRECTORY="$1"
    local MERGED_VIDEO_PATH="$DIRECTORY/merged_output.mp4"

    # Ensure directory exists
    if [ ! -d "$DIRECTORY" ]; then
        log_error "Directory '$DIRECTORY' does not exist"
        return 1
    fi

    # Find video files in the directory and sort by name
    local VIDEO_FILES=()
    while IFS= read -r -d '' file; do
        if is_video_file "$file"; then
            VIDEO_FILES+=("$file")
        fi
    done < <(find "$DIRECTORY" -maxdepth 1 -type f -print0 | sort -z)

    local VIDEO_COUNT=${#VIDEO_FILES[@]}

    if [ "$VIDEO_COUNT" -lt 2 ]; then
        log_warn "Found fewer than 2 video files in directory '$DIRECTORY', skipping merge."
        return 0
    fi

    log_info "Found $VIDEO_COUNT video files in directory '$DIRECTORY' for merging"

    # Create temporary file list
    local TEMP_FILE_LIST="$DIRECTORY/file_list.txt"
    > "$TEMP_FILE_LIST"
    
    for video in "${VIDEO_FILES[@]}"; do
        # Use absolute path and correctly handle special characters
        echo "file '$(realpath "$video")'" >> "$TEMP_FILE_LIST"
    done

    # Use ffmpeg's concat demuxer for merging
    log_info "Merging video files..."
    if ffmpeg -hide_banner -f concat -safe 0 -i "$TEMP_FILE_LIST" -c copy "$MERGED_VIDEO_PATH" >/dev/null 2>&1; then
        log_success "Merged $VIDEO_COUNT videos in directory '$DIRECTORY' into '$MERGED_VIDEO_PATH'"
    else
        log_error "Merging videos in directory '$DIRECTORY' failed."
    fi

    # Delete temporary file list
    rm -f "$TEMP_FILE_LIST"
}

# Display usage guide
show_usage() {
    echo "=============================================="
    echo "          Video Processing Script Usage Guide"
    echo "=============================================="
    echo "Usage: $0 <command> /path/to/video_or_directory [options]"
    echo ""
    echo "Command Descriptions:"
    echo "  getframe      Extract video frames"
    echo "  getmp4        Convert non-MP4 videos to MP4 format"
    echo "  mergevideo    Merge video files in a directory (processes each subdirectory recursively)"
    echo ""
    echo "Parameter Descriptions:"
    echo "  /path/to/video_or_directory  Specify a single video file or a directory containing video files."
    echo ""
    echo "Options:"
    echo "  -f, --fps <framerate>      Framerate for extracting frames (default is 1)"
    echo "  -h, --help                 Display this help message"
    echo ""
    echo "Supported Video Formats:"
    echo "  mp4, avi, mkv, mov, flv, wmv, webm"
    echo "=============================================="
}

# Process input directory or file (extract frames)
process_getframe() {
    local path="$1"
    local fps="${2:-1}"  # Default to 1
    
    if [ ! -e "$path" ]; then
        log_error "Path '$path' does not exist."
        return 1
    fi

    if [ -d "$path" ]; then
        log_info "Detected directory: $path"
        log_info "Starting recursive processing of video files in the directory and its subdirectories..."
        local video_files=()
        local count=0

        # Find all video files
        while IFS= read -r -d '' file; do
            if is_video_file "$file"; then
                video_files+=("$file")
                ((count++))
            fi
        done < <(find "$path" -type f -print0)

        log_info "Found $count video files in directory '$path' and its subdirectories"
        
        if [ $count -eq 0 ]; then
            log_warn "No video files found, exiting processing."
            return 0
        fi

        # Process each video file
        for file in "${video_files[@]}"; do
            log_info "Processing video file: $file"
            process_video "$file" "$fps"
        done
        
        log_success "All video files processed."
    elif [ -f "$path" ]; then
        if is_video_file "$path"; then
            log_info "Processing single video file: $path"
            process_video "$path" "$fps"
            log_success "Video file processing complete."
        else
            log_error "File '$path' is not a supported video format."
            return 1
        fi
    else
        log_error "'$path' is neither a file nor a directory."
        return 1
    fi
}

# Process input directory or file (convert to MP4)
process_getmp4() {
    local path="$1"
    
    if [ ! -e "$path" ]; then
        log_error "Path '$path' does not exist."
        return 1
    fi
    
    if [ -d "$path" ]; then
        log_info "Detected directory: $path"
        log_info "Starting conversion of non-MP4 video files in the directory..."
        local video_files=()
        local count=0

        # Find all non-MP4 video files
        while IFS= read -r -d '' file; do
            if is_video_file "$file" && [[ "${file,,}" != *".mp4" ]]; then
                video_files+=("$file")
                ((count++))
            fi
        done < <(find "$path" -type f -print0)

        log_info "Found $count non-MP4 video files in directory '$path' and its subdirectories"
        
        if [ $count -eq 0 ]; then
            log_warn "No non-MP4 video files found, exiting processing."
            return 0
        fi

        # Process each video file
        for file in "${video_files[@]}"; do
            log_info "Converting video file: $file"
            convert_to_mp4 "$file"
        done
        
        log_success "All video file conversions complete."
    elif [ -f "$path" ]; then
        if is_video_file "$path"; then
            convert_to_mp4 "$path"
        else
            log_error "File '$path' is not a supported video format."
            return 1
        fi
    else
        log_error "'$path' is neither a file nor a directory."
        return 1
    fi
}

# Process input directory (merge videos)
process_mergevideo() {
    local path="$1"
    
    if [ ! -e "$path" ]; then
        log_error "Path '$path' does not exist."
        return 1
    fi
    
    if [ -d "$path" ]; then
        log_info "Detected directory: $path"
        log_info "Starting recursive merging of video files in the directory..."
        
        # Get all subdirectories
        local directories=()
        while IFS= read -r -d '' dir; do
            directories+=("$dir")
        done < <(find "$path" -type d -print0)
        
        local dir_count=${#directories[@]}
        log_info "Found $dir_count directories in '$path'"
        
        # Iterate through each directory and merge videos
        for dir in "${directories[@]}"; do
            log_info "Processing directory: $dir"
            merge_videos_in_directory "$dir"
        done
        
        log_success "Merging of video files in all directories complete."
    elif [ -f "$path" ]; then
        log_error "'mergevideo' command requires a directory as input."
        return 1
    else
        log_error "'$path' is neither a file nor a directory."
        return 1
    fi
}

# Main program
main() {
    # Check number of arguments
    if [ $# -lt 1 ]; then
        show_usage
        exit 1
    fi
    
    # Parse command
    local COMMAND="$1"
    shift
    
    # Parse options
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
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                if [ -z "$INPUT_PATH" ]; then
                    INPUT_PATH="$1"
                else
                    log_error "Superfluous argument: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Check required parameters
    if [ -z "$INPUT_PATH" ]; then
        log_error "Missing path parameter"
        show_usage
        exit 1
    fi
    
    # Check for required commands
    if ! command -v ffmpeg >/dev/null 2>&1; then
        log_error "ffmpeg command not found. Please install ffmpeg."
        exit 1
    fi
    
    # Execute corresponding command
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
            log_error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main program
main "$@"
