#coding=utf-8
import os
import sys
import random
 
import numpy as np
import tensorflow as tf
# process a xml file
import xml.etree.ElementTree as ET
 
DIRECTORY_ANNOTATIONS = 'C:\\Users\\asus\\Desktop\\Annotations\\'
DIRECTORY_IMAGES = 'C:\\Users\\asus\\Desktop\\Annotations\\JPEGImages\\'
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 2000
 
VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}
 
def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
    values: A scalar or list of values.
    Returns:
    a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
 
def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
 
def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
 
SPLIT_MAP = ['train', 'val', 'trainval']
    
"""
Process a image and annotation file.
Args:
    filename:       string, path to an image file e.g., '/path/to/example.JPG'.
    coder:          instance of ImageCoder to provide TensorFlow image coding utils.
Returns:
    image_buffer:   string, JPEG encoding of RGB image.
    height:         integer, image height in pixels.
    width:          integer, image width in pixels.
读取一个样本图片及对应信息
"""
def _process_image(directory, name):
    # Read the image file.
    filename = os.path.join(directory, DIRECTORY_IMAGES, name + '.jpg')
    image_data = tf.gfile.FastGFile(filename, 'r').read()
    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()
    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text), int(size.find('width').text), int(size.find('depth').text)]
    # Find annotations.
    # 获取每个object的信息
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))
 
        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)
 
        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated
 
"""
Build an Example proto for an image example.
Args:
  image_data: string, JPEG encoding of RGB image;
  labels: list of integers, identifier for the ground truth;
  labels_text: list of strings, human-readable labels;
  bboxes: list of bounding boxes; each box is a list of integers;
      specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
      to the same label as the image label.
  shape: 3 integers, image shapes in pixels.
Returns:
  Example proto
将一个图片及对应信息按格式转换成训练时可读取的一个样本
"""
def _convert_to_example(image_data, labels, labels_text, bboxes, shape, difficult, truncated):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned
 
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example
 
"""
Loads data from image and annotations files and add them to a TFRecord.
Args:
  dataset_dir: Dataset directory;
  name: Image name to add to the TFRecord;
  tfrecord_writer: The TFRecord writer to use for writing.
"""
def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, 
                                  labels,
                                  labels_text,
                                  bboxes, 
                                  shape, 
                                  difficult, 
                                  truncated)
    tfrecord_writer.write(example.SerializeToString())
 
"""
以VOC2012为例，下载后的文件名为：VOCtrainval_11-May-2012.tar，解压后
得到一个文件夹：VOCdevkit
voc_root就是VOCdevkit文件夹所在的路径
在VOCdevkit文件夹下只有一个文件夹：VOC2012，所以下边参数year该文件夹的数字部分。
在VOCdevkit/VOC2012/ImageSets/Main下存放了20个类别，每个类别有3个的txt文件：
*.train.txt存放训练使用的数据
*.val.txt存放测试使用的数据
*.trainval.txt是train和val的合集
所以参数split只能为'train', 'val', 'trainval'之一
"""
def run(voc_root, year, split, output_dir, shuffling=False):
    # 如果output_dir不存在则创建
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
    # VOCdevkit/VOC2012/ImageSets/Main/train.txt
    # 中存放有所有20个类别的训练样本名称，共5717个
    split_file_path = os.path.join(voc_root,'VOC%s'%year,'ImageSets','Main','%s.txt'%split)
    print('>> ', split_file_path)
    with open(split_file_path) as f:
        filenames = f.readlines()
    # shuffling == Ture时，打乱顺序
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)
    # Process dataset files.
    i = 0
    fidx = 0
    dataset_dir = os.path.join(voc_root, 'VOC%s'%year)
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = '%s/%s_%03d.tfrecord' % (output_dir, split, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()
                filename = filenames[i].strip()
                _add_to_tfrecord(dataset_dir, filename, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
    print('\n>> Finished converting the Pascal VOC dataset!')


""""
if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('>> error. format: python *.py split_name')
    split = sys.argv[1]
    if split not in SPLIT_MAP:
        raise ValueError('>> error. split = %s' % split)
    run('./VOCdevkit', 2012, split, './')
""""

#原数据集路径，输出路径以及输出文件名
dataset_dir="C:\\Users\\asus\\Desktop\\VOC\\"
output_dir="C:\\Users\\asus\\Desktop\\tfrecords"
name="voc_train"
def main(_):
     run(dataset_dir, output_dir,name)
 
if __name__ == '__main__':
    tf.app.run()