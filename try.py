import shutil
import os


# 指定目录路径
folder_path = "result_save\gnn_2025-03-14-18-49-10\GNNExplainer"

# 允许保留的文件夹名称列表
allowed_folders = {"folder1", "folder2", "folder3"}  # 这里填入要保留的文件夹名称

# 获取当前目录下的所有文件夹
existing_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
print(existing_folders)


# 遍历并删除不在 `allowed_folders` 里的文件夹
for folder in existing_folders:
    folder_full_path = os.path.join(folder_path, folder)
    if folder not in allowed_folders:
        shutil.rmtree(folder_full_path)  # 递归删除文件夹
        print(f"已删除: {folder_full_path}")
