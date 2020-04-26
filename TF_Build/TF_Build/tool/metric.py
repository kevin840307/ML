import numpy as np
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def round_fix(value):
    round_pow = np.power(10, 2)
    return int(value * round_pow) / round_pow

def IOU(pred_image, label_image, num_class=2):
    # logical_and and logical_or only with binaray class
    # A * B / A + B - A * B

    IOUs = np.zeros(shape=(num_class))
    for index in range(num_class):
        Intersection = np.sum((label_image == index) * (pred_image == index))
        Union = np.sum(label_image == index) + np.sum(pred_image == index) - Intersection
        IOUs[index] = np.mean(Intersection[Union > 0] / Union[Union > 0])
        if math.isnan(IOUs[index]):
            IOUs[index] = 0
    result = np.mean(IOUs)
    return result

def BB_IOU(boxA, boxB):
    cx1,cy1,cx2,cy2 = boxA
    gx1,gy1,gx2,gy2 = boxB

    S_rec1 = (cx2 - cx1) * (cy2 - cy1)
    S_rec2 = (gx2 - gx1) * (gy2 - gy1)
 
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
 
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h 
 
    iou = area / (S_rec1 + S_rec2 - area)
    return iou

def Dice(pred, label):
    return np.sum(pred * label) * 2.0 / (np.sum(pred) + np.sum(label) + 1e-12)

def Dice2(pred, label):
    tn = TN(pred, label)
    tp = TP(pred, label)
    fp = FP(pred, label)
    return (tp * 2) / ((tp * 2) + tn + fp + 1e-12)

def TP(pred, label):
    return np.count_nonzero(pred * label)

def TN(pred, label):
    return np.count_nonzero((pred - 1) * (label - 1))

def FP(pred, label):
    return np.count_nonzero(pred * (label - 1))

def FN(pred, label):
    return np.count_nonzero((pred - 1) * label)

def precision(pred, label):
    tp = TP(pred, label)
    fp = FP(pred, label)
    result = tp / (tp + fp + 1e-12)
    return result

def recall(pred, label):
    tp = TP(pred, label)
    fn = FN(pred, label)
    result = tp / (tp + fn + 1e-12)
    return result

def f1(pred, label):
    P = precision(pred, label)
    R = recall(pred, label)
    result = 2. * P * R / (P + R + 1e-12)
    return result

def accuracy(pred, label):
    tp = TP(pred, label)
    fp = FP(pred, label)
    tn = TN(pred, label)
    fn = FN(pred, label)
    result = (tp + tn) / (tp + fp + tn + fn + 1e-12)
    return result
    #distance = np.subtract(label_image, pred_image)
    #distance = np.abs(distance)
    #result = 1. - np.mean(distance)
    #return result

def Jaccard(pred, label):
    return np.sum(pred * label) / (np.sum(pred) + np.sum(label) - np.sum(pred * label) + 1e-12)
