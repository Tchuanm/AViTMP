import torch
from torchvision.ops.boxes import box_area
import numpy as np


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = torch.tensor(x).unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x1, y1, w, h = torch.tensor(x).unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)

def box_xywh_to_xywh(x):
    x1, y1, w, h = torch.tensor(x).unbind(-1)
    b = [x1, y1, w, h]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):

    boxes1 = box_xywh_to_xyxy(boxes1).unsqueeze(0)
    boxes2 = box_xywh_to_xyxy(boxes2).unsqueeze(0)
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def center_distence(boxes1, boxes2):
    boxes1 = box_xywh_to_xywh(boxes1)
    boxes2 = box_xywh_to_xywh(boxes2)
    res = boxes1 - boxes2
    dist = torch.sqrt(res[0]**2 + res[1]**2)
    scale =  boxes1[2]*boxes1[3] / (boxes2[2]*boxes2[3])
    return dist, scale