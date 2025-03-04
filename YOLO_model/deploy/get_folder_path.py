import os
def get_script_directory():
    script_path = os.path.abspath(__file__)
    directory = os.path.dirname(script_path)
    print(f"脚本目录: {directory}")
    return directory
