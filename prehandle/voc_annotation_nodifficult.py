import xml.etree.ElementTree as ET
from os import getcwd
import os
sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["hat","person"]


def convert_annotation(year, image_id, list_file):
    in_file = open('../VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id),encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()
    a=1
    i=a
    
    for obj in root.iter('item'):
        #difficult = obj.find('').text
        cls = obj.find('name').text
        #if cls not in classes or int(difficult)==1:
            #continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        #print(image_id)
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    for obj in root.iter('object'):#找xml文档里的object 
        try:
            difficult = obj.find('difficult').text#找object对里的difficult对  在不同格式里可能找不到
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            #print(image_id)
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
            
        except Exception:
            print(i,end="")
    
   
    
            
        
      

wd = getcwd()

path1 = os.path.join(os.path.dirname(os.getcwd()), "")
print(path1)
for year, image_set in sets:
    image_ids = open('../VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set),encoding='utf-8').read().strip().split()
    #路径
    list_file = open('../model_data/%s_%s.txt'%(year, image_set), 'w',encoding='utf-8')
    for image_id in image_ids:
        list_file.write('%sVOCdevkit/VOC%s/JPEGImages/%s.jpg' %(path1, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
