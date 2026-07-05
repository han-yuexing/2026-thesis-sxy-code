# -*- coding: utf-8 -*-
'''
@File    :   visualizer.py
@Time    :   2022/04/05 11:39:33
@Author  :   Shilong Liu 
@Contact :   liusl20@mail.tsinghua.edu.cn; slongliu86@gmail.com
Modified from COCO evaluator
'''

import os, sys
from textwrap import wrap
import torch
import numpy as np
import cv2
import datetime

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils
from matplotlib import transforms

def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim() 
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        img_perm = img.permute(1,2,0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2,0,1)
    else: # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (img.size(1), str(img.size()))
        img_perm = img.permute(0,2,3,1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0,3,1,2)

class ColorMap():
    def __init__(self, basergb=[255,255,0]):
        self.basergb = np.array(basergb)
    def __call__(self, attnmap):
        # attnmap: h, w. np.uint8.
        # return: h, w, 4. np.uint8.
        assert attnmap.dtype == np.uint8
        h, w = attnmap.shape
        res = self.basergb.copy()
        res = res[None][None].repeat(h, 0).repeat(w, 1) # h, w, 3
        attn1 = attnmap.copy()[..., None] # h, w, 1
        res = np.concatenate((res, attn1), axis=-1).astype(np.uint8)
        return res

class COCOVisualizer():
    def __init__(self) -> None:
        pass

    def visualize(self, img, tgt, caption=None, dpi=120, savedir=None, show_in_console=True):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'
        """
        plt.figure(dpi=dpi)
        plt.rcParams['font.size'] = '10'
        ax = plt.gca()
        img = renorm(img).permute(1, 2, 0)
        ax.imshow(img)
        
        self.addtgt(tgt)
        if show_in_console:
            plt.show()

        if savedir is not None:
            if caption is None:
                savename = '{}/{}-{}.png'.format(savedir, int(tgt['image_id']), str(datetime.datetime.now()).replace(' ', '-'))
            else:
                savename = '{}/{}-{}-{}.png'.format(savedir, caption, int(tgt['image_id']), str(datetime.datetime.now()).replace(' ', '-'))
            print("savename: {}".format(savename))
            os.makedirs(os.path.dirname(savename), exist_ok=True)
            plt.savefig(savename)
        plt.close()

    #vision
    def addtgt(self, tgt):
        """
        - tgt: dict. args:
            - boxes: num_boxes, 4. xywh, [0,1].
            - box_label: num_boxes.
        """
        assert 'boxes' in tgt
        ax = plt.gca()
        H, W = tgt['size'].tolist() 
        numbox = tgt['boxes'].shape[0]

        # 定义固定的颜色映射：类别 0 对应绿色，类别 1 对应红色
        class_colors = {
            "crop": [0.0, 1.0, 0.0],  # 绿色 (RGB: [0, 1, 0])
            "weed": [1.0, 0.0, 0.0]   # 红色 (RGB: [1, 0, 0])
        }

        # class_colors={
        #     "Kena_(Commplina_benghalensio)": [1.0, 0.0, 0.0],       # 红
        #     "Lavhala_(Cyperus_Rotundus)": [1.0, 0.5, 0.0],          # 橙
        #     "Lamber_Quarter_plant_(Chenopodium)": [1.0, 1.0, 0.0],  # 黄
        #     "Little_Mallow_(Malva_parviflora)": [1.0, 0.0, 0.5],    # 粉红
        #     "Moti_dudhi_(Euphorbia_geneculata_L)": [0.5, 0.0, 1.0], # 紫
        #     "Obscure_morning_glory_(Ipomoea_obscura)": [0.0, 0.0, 1.0], # 蓝
        #     "Asian_Pigeonwings_(Clitoria_Ternatea)": [0.0, 1.0, 0.0],  # 作物绿
        #     "Bilayat_(Mexicana_Argemone)": [0.0, 1.0, 1.0],         # 青
        #     "Choti_dudhi_(Euphorbia_hirta)": [1.0, 0.75, 0.8],      # 浅粉
        #     "Digitaria_SP_(Digitaria_Sanguinalis)": [0.6, 0.3, 0.0], # 棕
        #     "Gajar_gavat_(Parthenium_hysterophorus)": [0.5, 0.5, 0.0], # 橄榄
        #     "Graceful_Sandmart_(Euphorbia_hypericifolia)": [0.75, 0.0, 0.25], # 紫红
        #     "Sicklepod_(Senna_obtusifolia)": [0.0, 0.5, 0.5],      # 深青
        #     "Harali_(Cynodon_dactylon)": [0.0, 1.0, 0.0],           # 作物绿
        #     "Dwarf_cassia_(Chamaecrista_pumila)": [0.0, 1.0, 0.0],  # 作物绿
        #     "Punarnava_(Boerhaavia_diffusa)": [0.0, 1.0, 0.0]       # 作物绿
        # }

        # class_colors = {
        #     "aeroplane": [0.0, 0.0, 1.0],      # 蓝色
        #     "bicycle": [0.0, 0.5, 0.5],        # 青色
        #     "bird": [1.0, 1.0, 0.0],           # 黄色
        #     "boat": [0.5, 0.0, 0.5],           # 紫色
        #     "bottle": [0.0, 1.0, 0.0],         # 绿色
        #     "bus": [1.0, 0.5, 0.0],            # 橙色
        #     "car": [1.0, 0.0, 0.0],            # 红色
        #     "cat": [0.0, 1.0, 1.0],            # 青色
        #     "chair": [0.5, 0.5, 0.0],          # 橄榄绿
        #     "cow": [0.0, 0.0, 0.5],            # 深蓝色
        #     "dining table": [0.5, 0.5, 0.5],   # 灰色
        #     "dog": [1.0, 0.0, 1.0],            # 粉红色
        #     "horse": [0.0, 0.5, 1.0],          # 天蓝色
        #     "motorbike": [1.0, 1.0, 1.0],      # 白色
        #     "person": [0.0, 0.0, 0.0],         # 黑色
        #     "potted plant": [0.5, 1.0, 0.5],   # 浅绿色
        #     "sheep": [0.5, 0.0, 1.0],          # 紫罗兰色
        #     "sofa": [1.0, 0.5, 0.5],           # 浅红色
        #     "train": [0.0, 0.5, 0.0],          # 深绿色
        #     "monitor": [0.5, 1.0, 1.0]      # 浅青色
        # }

        color = []
        polygons = []
        boxes = []
        for idx, box in enumerate(tgt['boxes'].cpu()):  # 使用 enumerate 获取索引 idx
            unnormbbox = box * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            polygons.append(Polygon(np_poly))

            # 获取当前框的类别标签
            label = tgt['box_label'][idx].item() if isinstance(tgt['box_label'], torch.Tensor) else tgt['box_label'][idx]

            # 根据类别标签选择颜色
            color.append(class_colors[label])

        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)

        if 'box_label' in tgt:
            assert len(tgt['box_label']) == numbox, f"Length mismatch: {len(tgt['box_label'])} != {numbox}"
            for idx, bl in enumerate(tgt['box_label']):
                _string = str(bl)
                bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
                ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

        if 'caption' in tgt:
            ax.set_title(tgt['caption'], wrap=True)

    #pre
    # def addtgt(self, tgt):
    #     """
    #     - tgt: dict. args:
    #         - boxes: num_boxes, 4. xywh, [0,1].
    #         - box_label: num_boxes.
    #         - scores: num_boxes. 置信度分数
    #     """
    #     assert 'boxes' in tgt
    #     ax = plt.gca()
    #     H, W = tgt['size'].tolist() 
    #     numbox = tgt['boxes'].shape[0]

    #     # # 定义固定的颜色映射：根据类别为每个框指定不同的颜色
    #     class_colors={
    #         "Kena_(Commplina_benghalensio)": [1.0, 0.0, 0.0],       # 红
    #         "Lavhala_(Cyperus_Rotundus)": [1.0, 0.5, 0.0],          # 橙
    #         "Lamber_Quarter_plant_(Chenopodium)": [1.0, 1.0, 0.0],  # 黄
    #         "Little_Mallow_(Malva_parviflora)": [1.0, 0.0, 0.5],    # 粉红
    #         "Moti_dudhi_(Euphorbia_geneculata_L)": [0.5, 0.0, 1.0], # 紫
    #         "Obscure_morning_glory_(Ipomoea_obscura)": [0.0, 0.0, 1.0], # 蓝
    #         "Asian_Pigeonwings_(Clitoria_Ternatea)": [0.0, 1.0, 0.0],  # 作物绿
    #         "Bilayat_(Mexicana_Argemone)": [0.0, 1.0, 1.0],         # 青
    #         "Choti_dudhi_(Euphorbia_hirta)": [1.0, 0.75, 0.8],      # 浅粉
    #         "Digitaria_SP_(Digitaria_Sanguinalis)": [0.6, 0.3, 0.0], # 棕
    #         "Gajar_gavat_(Parthenium_hysterophorus)": [0.5, 0.5, 0.0], # 橄榄
    #         "Graceful_Sandmart_(Euphorbia_hypericifolia)": [0.75, 0.0, 0.25], # 紫红
    #         "Sicklepod_(Senna_obtusifolia)": [0.0, 0.5, 0.5],      # 深青
    #         "Harali_(Cynodon_dactylon)": [0.0, 1.0, 0.0],           # 作物绿
    #         "Dwarf_cassia_(Chamaecrista_pumila)": [0.0, 1.0, 0.0],  # 作物绿
    #         "Punarnava_(Boerhaavia_diffusa)": [0.0, 1.0, 0.0]       # 作物绿
    #     }
    #     # class_colors = {
    #     #     "crop": [0.0, 1.0, 0.0],  # 绿色 (RGB: [0, 1, 0])
    #     #     "weed": [1.0, 0.0, 0.0]   # 红色 (RGB: [1, 0, 0])
    #     # }

    #     color = []
    #     polygons = []
    #     boxes = []
    #     for idx, box in enumerate(tgt['boxes'].cpu()):  # 使用 enumerate 获取索引 idx
    #         unnormbbox = box * torch.Tensor([W, H, W, H])  # 将归一化的坐标转换回图像尺寸
    #         unnormbbox[:2] -= unnormbbox[2:] / 2  # 计算左上角坐标
    #         [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()  # 获取坐标
    #         boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])

    #         # 为框创建多边形对象
    #         poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
    #         np_poly = np.array(poly).reshape((4, 2))
    #         polygons.append(Polygon(np_poly))

    #         # 获取当前框的类别标签
    #         label = tgt['box_label'][idx].item() if isinstance(tgt['box_label'], torch.Tensor) else tgt['box_label'][idx]

    #         # 获取当前框的置信度分数
    #         score = tgt['scores'][idx].item() if isinstance(tgt['scores'], torch.Tensor) else tgt['scores'][idx]

    #         # 根据类别标签选择颜色
    #         color.append(class_colors.get(label, [0.5, 0.5, 0.5]))  # 如果没有类别，使用灰色

    #         # 在框旁边添加置信度文本
    #         label_text = f'{label}: {score:.2f}'  # 格式化标签文本，显示类别和置信度
    #         ax.text(bbox_x, bbox_y, label_text, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

    #     # 创建颜色填充的多边形集合
    #     p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
    #     ax.add_collection(p)
    #     # 创建边框的多边形集合
    #     p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    #     ax.add_collection(p)

    #     # 如果包含类别标签，则在框内显示标签
    #     if 'box_label' in tgt:
    #         assert len(tgt['box_label']) == numbox, f"Length mismatch: {len(tgt['box_label'])} != {numbox}"
    #         for idx, bl in enumerate(tgt['box_label']):
    #             _string = str(bl)
    #             bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
    #             ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

    #     # 如果有标题（caption），则设置标题
    #     if 'caption' in tgt:
    #         ax.set_title(tgt['caption'], wrap=True)

    # def addtgt(self, tgt):
    #     """
    #     - tgt: dict. args:
    #         - boxes: num_boxes, 4. xywh, [0,1].
    #         - box_label: num_boxes.
    #     """
    #     assert 'boxes' in tgt
    #     ax = plt.gca()
    #     H, W = tgt['size'].tolist() 
    #     numbox = tgt['boxes'].shape[0]

    #     color = []
    #     polygons = []
    #     boxes = []
    #     for box in tgt['boxes'].cpu():
    #         unnormbbox = box * torch.Tensor([W, H, W, H])
    #         unnormbbox[:2] -= unnormbbox[2:] / 2
    #         [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
    #         boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
    #         poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
    #         np_poly = np.array(poly).reshape((4,2))
    #         polygons.append(Polygon(np_poly))
    #         c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
    #         color.append(c)

    #     p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
    #     ax.add_collection(p)
    #     p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    #     ax.add_collection(p)


    #     if 'box_label' in tgt:
    #         assert len(tgt['box_label']) == numbox, f"{len(tgt['box_label'])} = {numbox}, "
    #         for idx, bl in enumerate(tgt['box_label']):
    #             _string = str(bl)
    #             bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
    #             # ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': 'yellow', 'alpha': 1.0, 'pad': 1})
    #             ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

    #     if 'caption' in tgt:
    #         ax.set_title(tgt['caption'], wrap=True)


    