import pandas as pd
import base64
from PIL import Image
import io
import os
from tqdm import tqdm
import concurrent.futures
import time

def save_single_image(row, output_dir):
    """
    保存单张图片
    
    Args:
        row: DataFrame的一行数据，包含图片ID和base64数据
        output_dir: 输出目录
    """
    try:
        image_id = str(row[0])
        image_data = row[1]
        
        # 解码base64数据
        if image_data.startswith('/9j/'):
            image_data = "data:image/jpeg;base64," + image_data
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        # 解码并保存图片
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # 确保输出格式为JPEG
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # 保存图片
        output_path = os.path.join(output_dir, f"{image_id}.jpg")
        img.save(output_path, "JPEG", quality=95)
        return True
        
    except Exception as e:
        print(f"\nError processing image {image_id}: {str(e)}")
        return False

def save_images_from_df(df, output_dir, batch_size=1000):
    """
    从DataFrame保存所有图片
    
    Args:
        df: 包含图片数据的DataFrame
        output_dir: 输出目录
        batch_size: 批处理大小
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化计数器
    total_images = len(df)
    success_count = 0
    error_count = 0
    
    # 使用tqdm创建进度条
    with tqdm(total=total_images, desc="Saving images") as pbar:
        # 分批处理图片
        for start_idx in range(0, total_images, batch_size):
            end_idx = min(start_idx + batch_size, total_images)
            batch_df = df.iloc[start_idx:end_idx]
            
            # 使用线程池并行处理批次数据
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(save_single_image, row, output_dir)
                    for _, row in batch_df.iterrows()
                ]
                
                # 处理结果
                for future in concurrent.futures.as_completed(futures):
                    if future.result():
                        success_count += 1
                    else:
                        error_count += 1
                    pbar.update(1)
            
            # 每批次完成后打印进度
            if (end_idx % (batch_size * 10)) == 0:
                print(f"\nProgress: {end_idx}/{total_images} images processed")
                print(f"Success: {success_count}, Errors: {error_count}")
    
    # 打印最终统计信息
    print("\nProcessing completed!")
    print(f"Total images: {total_images}")
    print(f"Successfully saved: {success_count}")
    print(f"Errors: {error_count}")

# 使用示例
if __name__ == "__main__":
    # 设置输出目录
    output_dir = "output_images"
    
    # 调用保存函数
    tsv_path = "/data/dataset/webqa/WebQA_data_first_release/WebQA_imgs_7z_chunks/imgs.tsv"
    file = pd.read_csv(tsv_path, sep='\t', header=None)
    start_time = time.time()
    save_images_from_df(file, output_dir)
    end_time = time.time()
    
    # 打印总耗时
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")

