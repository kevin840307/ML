from __future__ import absolute_import


import scipy.io
import os
import numpy as np
import skimage.io as io
from skimage.transform import resize
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import selectivesearch

import sys, os
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
import metric as metric

ROI_THRESHOLD = 0.7

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def label_string_to_num(name):
    return CLASSES.index(name)

def label_num_to_string(index):
    return CLASSES[index]

def read_split_data(path):
    with open(path, "r") as fp:
        all_lines = fp.read()
        all_lines = str(all_lines).split('\n')
        return all_lines

def bb_norm(width, height, rect):
    if len(rect.shape) == 2:
        rect[:, 0] /=  width
        rect[:, 1] /=  height
        rect[:, 2] /=  width
        rect[:, 3] /=  height
    else:
        rect[0] /=  width
        rect[1] /=  height
        rect[2] /=  width
        rect[3] /=  height
    return rect

def bb_denorm(width, height, rect):
    if len(rect.shape) == 2:
        rect[:, 0] *=  width
        rect[:, 1] *=  height
        rect[:, 2] *=  width
        rect[:, 3] *=  height
    else:
        rect[0] *=  width
        rect[1] *=  height
        rect[2] *=  width
        rect[3] *=  height
    return rect

def bbox_transform(rois, inv_bboxs):
    bboxs = []
    for roi, inv_bbox in zip(rois, inv_bboxs):
        x1, y1, x2, y2 = roi
        width = x2 - x1
        height = y2 - y1
        x = (x1 + x2) / 2.
        y = (y1 + y2) / 2.

        rate_x1, rate_y1, rate_width, rate_height = inv_bbox
        inv_x = x + rate_x1 * width
        inv_y = y + rate_y1 * height
        inv_width = (width * np.exp(rate_width)) / 2.
        inv_height = (height * np.exp(rate_height)) / 2.
        bboxs.append([inv_x - inv_width, inv_y - inv_height, inv_x + inv_width, inv_y + inv_height])
    return np.array(bboxs)

def bbox_transform_inv(rois, gt_bboxs):
    inv_bboxs = []
    for roi, gt_bbox in zip(rois, gt_bboxs):
        x1, y1, x2, y2 = roi
        width = x2 - x1
        height = y2 - y1
        x = (x1 + x2) / 2.
        y = (y1 + y2) / 2.

        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1
        gt_x = (gt_x1 + gt_x2) / 2.
        gt_y = (gt_y1 + gt_y2) / 2.

        inv_bboxs.append([(gt_x - x) / width,
                      (gt_y - y) / height,
                      np.log(gt_width / width + 1e-8),
                      np.log(gt_height / height + 1e-8)])
    return np.array(inv_bboxs)

def one_hot_bboxs(labels):
    one_hot_bboxs = np.zeros((len(labels), (len(CLASSES) - 1) * 4 + 1))
    index = -1                         
    for label in labels:
        index += 1
        bbox = label[0:4]
        class_label = int(label[-1])
        if class_label == 0:
            continue
        one_hot_bboxs[index, (class_label - 1) * 4: class_label * 4] = bbox
        one_hot_bboxs[index, -1] = class_label
    return one_hot_bboxs

def one_hot_bbox_visualization(img, bboxs, labels, background=False, scale=False):
    if img.max() <= 1:
        img = img * 255

    img += np.array([[[102.9801, 115.9465, 122.7717]]])
    img = img.astype(np.uint8)
    image = Image.fromarray(img)
    width = image.width
    height = image.height
    draw = ImageDraw.Draw(image)

    for bbox, name_index in zip(bboxs, labels):
        name_index = int(name_index)
        if not background and name_index == 0:
            continue
        
        if scale:
            x1, y1, x2, y2 = bb_denorm(width, height, bbox[(name_index - 1) * 4: name_index * 4])
        else:
            x1, y1, x2, y2 = bbox[(name_index - 1) * 4: name_index * 4]

        print([x1, y1, x2, y2])
        draw.rectangle((x1, y1, x2, y2), outline='red', width=5)
        font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 24)
        draw.text((x1 + 5, y1 + 5), label_num_to_string(name_index), font=font)
    return image

def bbox_visualization(img, bboxs, labels, background=False, scale=False):
    if img.max() <= 1:
        img = img * 255
    img += np.array([[[102.9801, 115.9465, 122.7717]]])
    img = img.astype(np.uint8)
    image = Image.fromarray(img)
    width = image.width
    height = image.height
    draw = ImageDraw.Draw(image)
    for label, bbox in zip(labels, bboxs):
        name_index = int(label)

        if not background and name_index == 0:
            continue
        
        if scale:
            x1, y1, x2, y2 = bb_denorm(width, height, bbox)
        else:
            x1, y1, x2, y2 = bbox

        draw.rectangle((x1, y1, x2, y2), outline='red', width=5)
        font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 24)
        draw.text((x1 + 5, y1 + 5), label_num_to_string(name_index), font=font)
    return image


def felzenszwalb(img, original_labels, scale=20, sigma=0.9, min_size=50):
    rois = []
    bboxs = []
    labels = []
    _, regions = selectivesearch.selective_search(np.array(img), scale=scale, sigma=sigma, min_size=min_size)
    width = np.shape(img)[1]
    height = np.shape(img)[0]
    background_max = 4
    background_count = 0

    for region in regions:
        x, y, h, w = region['rect']
        label = [0, 0, 0, 0, 0]

        for original_label in original_labels:
            x1, y1, x2, y2, class_label = original_label
            if metric.BB_IOU((x1, y1, x2, y2), (x, y, x + h, y + w)) >= ROI_THRESHOLD:
                label = original_label

        if label[-1] == 0 and (h < 7 or w < 7 or
                           background_count > background_max or
                           np.random.randint(0, 99) > 2):
            continue

        if label[-1] == 0:
            background_count += 1

        #roi = bb_norm(width, height, [x, y, (x + h), (y + w)])
        roi = (x, y, (x + h), (y + w))
        rois.append(roi)
        labels.append(label)
    return np.array(rois), np.array(labels)

def read_img(img_path, xml_path, selectivesearch=False):
    tree = etree.parse(xml_path)
    img = Image.open(img_path)
    width = img.width
    height = img.height

    scale = 800. / width
    re_image = img.resize((800, int(scale * height)))
    re_image = np.array(re_image)

    img = np.array(img)
    labels = []
    rois = []
    for region in tree.xpath("//annotation//object"):
        name = region.xpath("./name")[0].text
        x1 = float(region.xpath("./bndbox//xmin")[0].text)
        y1 = float(region.xpath("./bndbox//ymin")[0].text)
        x2 = float(region.xpath("./bndbox//xmax")[0].text)
        y2 = float(region.xpath("./bndbox//ymax")[0].text)
        labels.append([x1, y1, x2, y2, label_string_to_num(name)])

    if selectivesearch:
        rois, labels = felzenszwalb(img, labels)

    labels = np.array(labels)
    rois = np.array(rois)

    rois = rois * scale
    labels[:, 0:4] = labels[:, 0:4] * scale

    #labels = one_hot_bboxs(labels)

    return re_image, rois, labels

class VOCDataset():
    def __init__(self, img_path='D:/下載/Faster-RCNN-TensorFlow-Python3-master/Faster-RCNN-TensorFlow-Python3-master/data/VOCdevkit2007/VOC2007'):
        self.img_path = img_path
        self.build()

    def build(self):
        train = read_split_data(self.img_path + '/ImageSets/Main/train.txt')
        self.train = train[0:len(train) - 1]

        validation = read_split_data(self.img_path + '/ImageSets/Main/trainval.txt')
        self.validation = train[0:len(train) - 1]

        test = read_split_data(self.img_path + '/ImageSets/Main/test.txt')
        self.test = train[0:len(train) - 1]


    def get_minbatch(self, batch_size, time, type='train', selectivesearch=True):
        images = []
        bboxs = []
        labels = []
        rois = []
        if type == 'train':
            data_index = np.copy(self.train)
        elif type == 'test':
            data_index = np.copy(self.test)
        else:
            data_index = np.copy(self.validation)

        echoe = (len(data_index) - batch_size) // batch_size
        start_index = (time % echoe) * batch_size
        np.random.seed((time // echoe) + 1)
        np.random.shuffle(data_index)

        for index in range(start_index , start_index + batch_size, 1):
            img_name = data_index[index]
            image, roi, label = read_img(self.img_path + '/JPEGImages/' + img_name + '.jpg', self.img_path + '/Annotations/' + img_name + '.xml', selectivesearch)
            image = image.astype(np.float) - np.array([[[102.9801, 115.9465, 122.7717]]])
             
            #if image.max() > 1:
            #    image = image / 255.

            images.append(image)
            rois.append(roi)
            labels.append(label)
            
        return np.array(images), np.array(labels), np.array(rois)

    def get_size(self, type='train'):
        if type == 'validation':
            return len(self.validation)
        if type == 'train':
            return len(self.train)
        return len(self.test)

if __name__ == '__main__':
    dataset = VOCDataset()
    dataset.build()
    images, rois, labels = dataset.get_minbatch(1, 0, selectivesearch=True)

    index = 221

    for image, roi, label in zip(images, rois, labels):
        label[:, 0:4] = bbox_transform_inv(roi, label[:, 0:4])
        print(label[:, 0:4])
        label[:, 0:4] = bbox_transform(roi, label[:, 0:4])
        print(label[:, 0:4])
        plt.subplot(index)
        img = bbox_visualization(image, label[:, 0:4], label[:, -1], scale=False)
        plt.imshow(img)
        index += 1


    plt.show()
    