import os
from icrawler.builtin import BingImageCrawler

# 定义要下载的类别及对应的搜索关键词（支持多语言）
categories = {
    "胡萝卜": ["carrot"],
    "白萝卜": ["daikon"],
    "鹅卵石": ["cobblestone","pebbles"],
    "砖块（碎的）": ["碎砖"],
    "土豆": ["potato","土豆"]
}

# 设置每个关键词下载的图片数量
num_images_per_keyword = 3000  # 

# 获取当前代码文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建保存图片的主目录（相对当前目录）
output_dir = os.path.join(current_dir, "yolo_dataset")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历每个类别并下载图片
for category, keywords in categories.items():
    print(f"正在下载类别 '{category}' 的图片...")
    # 创建类别对应的文件夹
    category_path = os.path.join(output_dir, category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)
    
    for keyword in keywords:
        print(f"  正在使用关键词 '{keyword}' 下载图片...")
        # 初始化 BingImageCrawler
        crawler = BingImageCrawler(
            feeder_threads=4,
            parser_threads=4,
            downloader_threads=16,
            storage={'root_dir': category_path}
        )
        
        # 开始抓取图片
        try:
            crawler.crawl(
                keyword=keyword,
                max_num=num_images_per_keyword,
                min_size=(200, 200),  # 可根据需要调整图片最小尺寸
                file_idx_offset=0
            )
            print(f"  关键词 '{keyword}' 的图片下载完成。")
        except Exception as e:
            print(f"  使用关键词 '{keyword}' 下载图片时出错: {e}")
    
    print(f"类别 '{category}' 的所有图片下载完成。\n")

print("所有类别的图片下载完成！")
