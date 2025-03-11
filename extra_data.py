"""
    将每个预处理过后的数据集图片及其对应的label写入csv中，并统计数据集的图片数量
"""


import os
import csv

# 配置参数
root_dir = r"E:\github_code\Unnamed1\dataset\processed"  # 数据集根目录
output_csv = r"E:\github_code\Unnamed1\global_labels.csv"  # 输出CSV文件路径
output_txt = r"E:\github_code\Unnamed1\different_dataset_photos_nums.txt"  # 输出TXT文件路径


def process_dataset(root_dir, output_csv, output_txt, current_dataset):
    # 检查CSV文件是否存在以及是否为空
    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0

    # 打开CSV文件并写入表头（如果文件为空）
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 如果文件不存在或为空，写入表头
        if not file_exists:
            writer.writerow(['path', 'label'])  # 写入列头
        else:
            print("CSV文件已存在，跳过写入列头。")

        # 初始化字典记录每个子目录下的图片数量
        photos_nums = {}

        # 遍历所有子目录 DFDC FF++
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    # 获取完整路径
                    full_path = os.path.join(root, file)

                    # 转换路径格式（保留从"dataset"开始的部分）
                    relative_path = full_path.split("dataset")[-1]
                    formatted_path = os.path.join("dataset", relative_path.strip("\\/")).replace("\\", "/")

                    # 根据父目录确定标签
                    if "original_sequences" in full_path:
                        label = "REAL"
                    elif "Celeb-real" in full_path:
                        label = "REAL"
                    elif "YouTube-real" in full_path:
                        label = "REAL"
                    elif "real" in full_path:
                        label = "REAL"
                    elif "original_video" in full_path:
                        label = "REAL"
                    elif "original_sequences" in full_path:
                        label = "REAL"
                    elif "fake" in full_path:
                        label = "FAKE"
                    elif "method_A" in full_path:
                        label = "FAKE"
                    elif "method_B" in full_path:
                        label = "FAKE"
                    elif "Celeb-synthesis" in full_path:
                        label = "FAKE"
                    elif "manipulated_sequences" in full_path:
                        label = "FAKE"
                    else:
                        label = "UNKNOWN"  # 安全兜底

                    # 写入CSV
                    writer.writerow([formatted_path, label])

                    # 更新子目录下的图片数量
                    subdir = os.path.relpath(root, root_dir)
                    if subdir == '.':
                        subdir = ''  # 如果是当前目录，则不添加任何子目录信息
                    if current_dataset == "FaceForensics++" :
                        subdir = subdir.split(os.sep)[:2]  # 获取下两级目录
                        subdir = os.sep.join(subdir)  # 重新组合目录路径
                    else:
                        subdir = subdir.split(os.sep)[0]  # 获取下一级目录
                    if subdir not in photos_nums:
                        photos_nums[subdir] = 0
                    photos_nums[subdir] += 1

    # 写入TXT文件 # 使用 'a' 模式进行追加写入
    with open(output_txt, 'a', encoding='utf-8') as txtfile:
        txtfile.write(f"{current_dataset}\n")
        for subdir, num in photos_nums.items():
            txtfile.write("   |____________")
            txtfile.write(f"{subdir:40} 数量为:{num}\n")
        print("\n")

    print(f"CSV文件已生成：{os.path.abspath(output_csv)}")
    print(f"TXT文件已生成：{os.path.abspath(output_txt)}")


# 数据集 没有DFDC 因为拿DFDC做测试集
# list = ['DFDC'] # 对DFDC测试集只做数量统计
list = ['Celeb-DF-v1','Celeb-DF-v2','DFDCP','UADFV','FaceForensics++']
for current_dataset in list:
    print(f"\n正在处理数据集{current_dataset}")
    current_dir = root_dir + '\\' + current_dataset
    print(current_dir)
    # 执行处理
    process_dataset(current_dir, output_csv, output_txt, current_dataset)