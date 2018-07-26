# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:28:59 2018

@author: asus
"""

import os
import tensorflow as tf
import xml.etree.ElementTree as ET
import sys
sys.path.append('C:\\Users\\asus\\Desktop')
from data2example import _convert_to_example


#读取图片
data_dir = 'C:\\Users\\asus\\Desktop\\1'
filenames = os.listdir(os.path.join(data_dir,'Annotations'))
for file in filenames:
    image_idx = file.split('.')[0]
    image_path = '%s\\JPEGImages\\%s.jpg'%(data_dir, image_idx)
    annotation_path = '%s\\Annotations\\%s.xml'%(data_dir, image_idx)
    # read file
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    print(type(image_data))

    #读取annotation文件
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    print(shape)

    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        print(label)
        labels.append(1)#int(VOC_LABELS[label][0]) label对应的类别编号， 此处直接使用1， 没什么特殊含义。
        labels_text.append(label.encode('utf-8'))
    
        if obj.find('difficult') is not None:
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
    
        if obj.find('truncated') is not None:
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)
    
        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))

    #将Example写入文件
    tf_filename = '%s\\tf\\%s.tfrecords'%(data_dir, image_idx)
    tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)
    example = _convert_to_example(image_data, labels, labels_text,  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())
