import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from scipy.spatial.distance import pdist


class IS(nn.Module):
    def __init__(self,weight_ce=0.5,weight_trv=0.5):

       
        self.weight_ce = weight_ce
        self.weight_trv = weight_trv
        self.ce = ISCE()


        self.trv=ISTV()
        self.weight_trv=weight_trv
    def forward(self, net_output, target):

        
        mask = mask.float()

        Isce = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0

        Istv = self.trv(net_output,target) if self.weight_trv != 0 else 0
  
        if self.aggregate == "sum":

            result = self.weight_ce * Isce + self.weight_trv * Istv
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class ISTV(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

  
        TV = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred * target).sum(dim=1).sum(dim=1).sum(dim=1)+
                                            0.2 * (pred * (1 - target)).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            0.8 * ((1 - pred) * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        
        TV_dis = torch.clamp((1 - TV).mean(), 0, 2)
    
        #防止为0导致nan以及无穷小-inf（小于0和接近于0）
       
        if TV_dis<0:
            TV_dis=1.00001
        else:
            TV_dis=TV_dis+1.00001
        if TV_dis > 50:
            TV_dis=50

        return torch.log(TV_dis)
    


class ISCE(nn.Module):

    def __init__(self, aggregate="sum",  weight_ce=1):
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = nn.CrossEntropyLoss()
    def forward(self, net_output, target):
        
        mask = mask.float()

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()
        
        if ce_loss<0:
            ce_loss=1.00001
        else:
            ce_loss=ce_loss+1.00001
        if ce_loss > 50:
            ce_loss=50

        return ce_loss


