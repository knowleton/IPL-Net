import os
from tqdm import tqdm
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

# import Config
# from models import models_3d
from utils.DataUtils import Dataset3D_Predict
from utils.MetricsUtils import binary_torch
from utils.LogUtils import get_month_and_day, get_log_fold_dir
from utils.AuxUtils import get_args, get_ckpt_path, predict_merge_channel_torch, set_pred_dir, get_index, muilt_sliding_window_inference_3d, save_predict_mask
from utils import Vnetlosses
import pandas as pd
from utils.LossUtils import Loss, dice_with_norm_binary
import os
import SimpleITK as sitk
import csv
import numpy as np
from utils.Vnetlosses import FocalLoss
def readImage(file_name):
    itk_image_ite1 = sitk.ReadImage(file_name)
    data_npy_ite1 = sitk.GetArrayFromImage(itk_image_ite1)
    return data_npy_ite1
def pathProcess(rootPath):
    nextFolder=os.listdir(rootPath)[0]
    csvPath=os.path.join(rootPath,nextFolder,f'fold0\第{i}迭代.csv')
    # csvPath=os.path.join(rootPath,nextFolder,'fold0\第3迭代.csv')
    labelsPathRoot=os.path.join(rootPath,nextFolder,'fold0')
    labelsNames=os.listdir(labelsPathRoot)



    return csvPath,labelsPathRoot,labelsNames
def readCsv(csvPath):

    # 创建两个空列表，用于存储name和consistency_loss
    names = []
    consistency_losses = []
    # 打开CSV文件
    with open(csvPath, 'r') as csvfile:
        # 创建CSV阅读器
        csvreader = csv.reader(csvfile)
        
        # 跳过标题行（如果CSV文件有标题行的话）
        next(csvreader)
        
        # 遍历CSV文件中的每一行
        for row in csvreader:
            #name切分，consistency转lf
            '''''data/NPCNS_Cutted/images/1708289P.nii.gz'''''
            name=row[0].split('/')[-1]
            consistency_loss=float(row[1])
            # 将name添加到names列表，将consistency_loss添加到consistency_losses列表
            names.append(name)
            consistency_losses.append(consistency_loss)
    return names,consistency_losses




def countConsistency(csvPaths):
    all_csvNames = []
    all_csvConsistency = []
    for csvPath in csvPaths:
        csvNames,csvConsistency=readCsv(csvPath)
        all_csvNames.append(csvNames)
        all_csvConsistency.append(csvConsistency)
    # 检查所有文件名是否相等
    # 获取第一个子列表的第一个元素作为基准
    first_element = all_csvNames[0][0]
    # 获取第一个子列表的长度作为基准长度
    first_length = len(all_csvNames[0])
    # 遍历所有子列表，检查第一个元素和长度是否与基准相等
    for sublist in all_csvNames:
        assert sublist[0] == first_element, f"第一个元素不匹配: {sublist[0]} != {first_element}"
        assert len(sublist) == first_length, f"长度不匹配: {len(sublist)} != {first_length}"
    
    # 计算平均一致性损失值
    average_consistency_losses = []
    for i in range(len(all_csvConsistency[0])):
        average = sum(loss[i] for loss in all_csvConsistency) / len(all_csvConsistency)
        average_consistency_losses.append(average)
    return average_consistency_losses, all_csvNames[0]
    
def countDiffence(labelsPathRoot,labelsNames):
    datanum=len(labelsNames)
    iterNum=len(labelsPathRoot)
    diffList=[]
    for i in range(datanum):
        labelname=labelsNames[i]
        iterlabelPaths=[]
        #为自适应迭代数，对标签遍历i，对迭代遍历j，将组织好的同一影像，不同
        #迭代路径编成列表，之后遍历列表直接计算前一位和后一位的diff，并平均（iterNum）即可
        for j in range(iterNum):
            itetlabelpath=os.path.join(labelsPathRoot[j],labelname)
            iterlabelPaths.append(itetlabelpath)
        #遍历迭代列表,计算伪标签不稳定度，并平均，并插入列表
        diffBuf=0.0
        for iterI in range(iterNum-1):
            diffBuf+=coutDiff(iterlabelPaths[iterI],iterlabelPaths[iterI+1])
        diffBuf=diffBuf/iterNum
        diffList.append(diffBuf)

    return diffList,labelsNames

#计算两个伪标签之间的不稳定性
def coutDiff(labelPath1,labelPath2):
    focalcouter=FocalLoss()
    itk_image_ite1 = sitk.ReadImage(labelPath1)
    data_npy_ite1 = sitk.GetArrayFromImage(itk_image_ite1)
    itk_image_ite2 = sitk.ReadImage(labelPath2)
    data_npy_ite2 = sitk.GetArrayFromImage(itk_image_ite2)
    
    uncertainty1 = np.sum(data_npy_ite1 != data_npy_ite2)/np.sum(data_npy_ite2>0)
    uncertainty2=focalcouter.forward(data_npy_ite1,data_npy_ite2)

    return uncertainty1+uncertainty2

if __name__ == "__main__":
    paslabelsPaths=['predicts/iter1',
               
'/predicts/iter2',

'predicts/iter3',

'predicts/iter4',

               ]
    csvPaths=[]#eg 迭代的csv路径
    labelsRootPaths=[]#eg 迭代的标签Root路径
    labelsNamesList=[]#eg 迭代分别的标签名字表，读文件的时候现拼路径
    for paslabelsPath in paslabelsPaths:
        # print(pathProcess(paslabelsPath))
        csvPath,labelsPathRoot,labelsNames=pathProcess(paslabelsPath)
        csvPaths.append(csvPath)
        labelsRootPaths.append(labelsPathRoot)
        labelsNamesList.append(labelsNames)
        # print(csvPath,labelsPathRoot)
    #检查是否有不同长度的
    for labelsNames in labelsNamesList:
        print(len(labelsNames))
    
    #得到每个数据与对应的模型平均不确定度
    # print(countConsistency(csvPaths))
    average_consistency_losses, datanamesConsis=countConsistency(csvPaths)
    average_Diff_losses,datanamesDiff=countDiffence(labelsPathRoot,labelsNames)

    print(average_consistency_losses,datanamesConsis)
    print(average_Diff_losses,datanamesDiff)
    
    #循环加权计算，并记录进表格
    resUncertaincy=[]
    for i in range(len(average_Diff_losses)):
        resBuf=average_consistency_losses[i]+average_Diff_losses[i]
        resUncertaincy.append(resBuf)


    # 创建一个空的DataFrame来保存结果
    result_df = pd.DataFrame(columns=['name','consistency_loss'])


    #记录进表格里
    result_df = pd.concat([result_df, pd.DataFrame({
        'name': [datanamesConsis],
        'resUncertaincy': [resUncertaincy]
    })], ignore_index=True)


    #存储不确定值表
    result_df.to_csv('不确定性表.csv', index=False)








    






        


