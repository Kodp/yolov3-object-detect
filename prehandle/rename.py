import os

# 设置初始目录
file_dir = r'D:/123'

for root,dirs,files in os.walk(file_dir):
    # 设置路径到每个子文件夹，子子文件夹......
    os.chdir(root)
    i = 1
    # 遍历每个子文件夹，子子文件夹......中的每个文件
    for filespath in files:
        # 将原本的文件的后缀名提取出来，先以‘.’进行分割，然后用old_file_name_split[-1]提取出后缀名
        old_file_name_split = filespath.split('.')
        # 将新名称修改为1.txt, 2.txt, ......
        new_name = "mp4"+str(i) + '.' + old_file_name_split[-1]
        # 替换名称（注意，原本的名称不能有1.txt等，不然会替换失败）
        os.rename(filespath, new_name)
        i += 1