# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/5/16$ 17:12$
# @Author  : KOD Chen
# @Email   : 821237536@qq.com
# @File    : configs$.py
# Description :anchor尺寸、coco数据集的80个classes类别名称
# --------------------------------------

anchors = [[0.57273, 0.677385],
           [1.87446, 2.06253],
           [3.33843, 5.47434],
           [7.88282, 3.52778],
           [9.77052, 9.16828]]

def read_coco_labels():
    f = open("./yolo2_data/coco_classes.txt")
    class_names = []
    for l in f.readlines():
        l = l.strip() # 去掉回车'\n'
        class_names.append(l)
    return class_names

class_names = read_coco_labels()
