import os
import shutil
import random

root_dir = '/hy-tmp/continual_learning_with_vit/data/NihAll1000'  # 替换为根目录的路径
output_dir = '/hy-tmp/continual_learning_with_vit/data/Nih/NihAll1000'  # 替换为输出目录的路径

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 遍历根目录下的子目录
for label_dir in os.listdir(root_dir):
    label_path = os.path.join(root_dir, label_dir)
    if os.path.isdir(label_path):
        # 检查是否是目录
        class_label = label_dir  # 类别标签就是子目录的名称

        # 遍历当前子目录下的图像文件
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            if os.path.isfile(file_path):
                # 检查是否是文件
                output_file_path = os.path.join(output_dir, file_name)
                shutil.copy(file_path, output_file_path)

# 获取输出目录下的图像文件列表
image_files = []
for image_file_name in os.listdir(output_dir):
    image_file_path = os.path.join(output_dir, image_file_name)
    if os.path.isfile(image_file_path):
        # 检查是否是文件
        image_files.append(image_file_path)
        
        
# 修复PNG文件名中存在两个点号的问题
for image_file_path in image_files:
    file_dir, file_name = os.path.split(image_file_path)
    new_file_name = file_name.replace('..png', '.png')
    new_image_file_path = os.path.join(file_dir, new_file_name)
    os.rename(image_file_path, new_image_file_path)

# 打乱图像文件列表的顺序
random.shuffle(image_files)

# 计算划分的索引位置
split_index = int(0.8 * len(image_files))

# 生成train.txt和test.txt文件
import os
import random

# 生成train.txt和test.txt文件
train_file = open('/hy-tmp/continual_learning_with_vit/data/Nih/train.txt', 'w')
test_file = open('/hy-tmp/continual_learning_with_vit/data/Nih/test.txt', 'w')
label_file = open('/hy-tmp/continual_learning_with_vit/data/Nih/labels.txt', 'w')

# 将图像文件添加到train.txt或test.txt
# 遍历根目录下的子目录
class_labels = {}  # 类别标签字典，用于记录类别与数字的对应关系
label_count = 0  # 当前类别计数器

for label_dir in os.listdir(root_dir):
    label_path = os.path.join(root_dir, label_dir)
    if os.path.isdir(label_path):
        # 检查是否是目录
        class_label = str(label_count)  # 数字代替类别标签
        class_labels[class_label] = label_dir  # 记录类别与数字的对应关系
        label_count += 1

        # 遍历当前子目录下的图像文件
        file_list = os.listdir(label_path)
        random.shuffle(file_list)  # 随机打乱文件列表

        train_size = int(len(file_list) * 0.8)  # 计算训练集大小
        train_files = file_list[:train_size]  # 前80%作为训练集
        test_files = file_list[train_size:]  # 后20%作为测试集

        # 写入train.txt
        for file_name in train_files:
            file_path = os.path.join(label_path, file_name)
            output_path = os.path.join(output_dir, file_name)
            if os.path.isfile(file_path):
                # 检查是否是文件
                file_line = f"{output_path} {class_label}\n"
                train_file.write(file_line)

        # 写入test.txt
        for file_name in test_files:
            file_path = os.path.join(label_path, file_name)
            output_path = os.path.join(output_dir, file_name)
            if os.path.isfile(file_path):
                # 检查是否是文件
                file_line = f"{output_path} {class_label}\n"
                test_file.write(file_line)

# 关闭文件
train_file.close()
test_file.close()

# 输出类别标签与数字的对应关系到labels.txt文件
for class_label, label_name in class_labels.items():
    label_line = f"{class_label} {label_name}\n"
    label_file.write(label_line)

# 关闭文件
label_file.close()
