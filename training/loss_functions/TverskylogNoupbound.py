"""

Tversky loss
"""

import torch
import torch.nn as nn


class TverskyLogLoss_noupbound(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # dice系数的定义
        dice = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred * target).sum(dim=1).sum(dim=1).sum(dim=1)+
                                            0.2 * (pred * (1 - target)).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            0.8 * ((1 - pred) * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是dice距离
        dice_dis = torch.clamp((1 - dice).mean(), 0, 2)
        # print(dice_dis)
        #防止为0导致nan以及无穷小-inf（小于0和接近于0）
        # dice_dis=dice_dis+2.7
        if dice_dis<0:
            dice_dis=1.0000001
        else:
            dice_dis=dice_dis+1.0000001
        # if dice_dis > 50:
        #     dice_dis=50

        return torch.log(dice_dis)
    

