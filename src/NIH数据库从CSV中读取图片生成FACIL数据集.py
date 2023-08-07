import os
import shutil

def rename_files(folder_path):
    # 遍历文件夹下的子文件夹及文件
    for root, dirs, files in os.walk(folder_path):
        # 遍历文件
        for file_name in files:
            # 获取原始文件路径
            old_file_path = os.path.join(root, file_name)
            # 去除空格、_、-字符，生成新的文件名,..png改为.png
            new_file_name = file_name.replace(" ", "").replace("_", "").replace("-", "").replace("..png",".png")
            # 生成新的文件路径
            new_file_path = os.path.join(root, new_file_name)
            
            if file_name != new_file_name:
                # 重命名文件并覆盖之前的文件
                shutil.move(old_file_path, new_file_path)
                print(f"已重命名文件：{file_name} -> {new_file_name}")

# 测试
folder_path = "/hy-tmp/continual_learning_with_vit/data/NihAll1000"
rename_files(folder_path)