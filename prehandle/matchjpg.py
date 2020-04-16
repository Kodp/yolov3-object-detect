#匹配图像 你在标label的时候可能有一些图片没标 就会和xml文档对不上 把这些不标的图片清除

import os
import numpy as np
# 设置初始目录
file_dir1 = r'D:\Safehat\yolo3-keras2-self\VOCdevkit\VOC2007\Annotations/'#xml文档目录
file_dir2 = r'D:\Safehat\yolo3-keras2-self\VOCdevkit\VOC2007\JPEGImages/'#图片目录 
listname=[]
for root,dirs,files in os.walk(file_dir1):
    # 设置路径到每个子文件夹，子子文件夹......
    os.chdir(root)
    i = 1
    
    # 遍历每个子文件夹，子子文件夹......中的每个文件
    for filespath in files:
        old_file_name_split = filespath.split('.') #得到xml文档的所有【前缀名 后缀名】
        listname.append(old_file_name_split)
        #print(old_file_name_split[1])   # 将原本的文件的后缀名提取出来，先以‘.’进行分割，然后用old_file_name_split[-1]提取出后缀名
        #print(old_file_name_split[0]) 
listname=np.array(listname)
listname=listname[:,0]#将前缀名变为列表【前缀名】
for i in listname:
    print(i)

for root1,dirs1,files1 in os.walk(file_dir2):
    for filespath1 in files1:
        os.chdir(root1)
        old_file_name_split1 = filespath1.split('.')
        #print(old_file_name_split1[0]+old_file_name_split1[1])
        
        if old_file_name_split1[0] not in listname:#【如果jpg的前缀名不存在xml前缀名列表里 删除】
            os.remove(file_dir2+str(old_file_name_split1[0])+'.jpg')
                
    #将新名称修改为1.txt, 2.txt, ......
    #new_name = "mp4"+str(i) + '.' + old_file_name_split[-1]
    #替换名称（注意，原本的名称不能有1.txt等，不然会替换失败）
    #os.rename(filespath, new_name)
    #i += 1
      