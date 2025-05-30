import os
from icrawler.builtin import BingImageCrawler

# Define categories to download and their corresponding search keywords (supports multiple languages)
categories = {
    "carrot": ["carrot"],
    "daikon": ["daikon"],
    "cobblestone": ["cobblestone", "pebbles"],
    "broken_bricks": ["broken bricks"],
    "potato": ["potato"]
}

# Set number of images to download per keyword
num_images_per_keyword = 3000

# Get current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create main output directory (relative to current directory)
output_dir = os.path.join(current_dir, "yolo_dataset")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Download images for each category
for category, keywords in categories.items():
    print(f"Downloading images for category '{category}'...")
    # Create category folder
    category_path = os.path.join(output_dir, category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)
    
    for keyword in keywords:
        print(f"  Downloading images with keyword '{keyword}'...")
        # Initialize BingImageCrawler
        crawler = BingImageCrawler(
            feeder_threads=4,
            parser_threads=4,
            downloader_threads=16,
            storage={'root_dir': category_path}
        )
        
        # Start crawling images
        try:
            crawler.crawl(
                keyword=keyword,
                max_num=num_images_per_keyword,
                min_size=(200, 200),  # Adjust minimum image size as needed
                file_idx_offset=0
            )
            print(f"  Image download completed for keyword '{keyword}'.")
        except Exception as e:
            print(f"  Error downloading images with keyword '{keyword}': {e}")
    
    print(f"All images downloaded for category '{category}'.\n")

print("Image download completed for all categories!")
