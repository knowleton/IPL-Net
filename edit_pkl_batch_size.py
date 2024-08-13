#https://blog.csdn.net/Joker00007/article/details/125588792
import numpy as np
import pickle as pkl
from batchgenerators.utilities.file_and_folder_operations import *

# path = r'/home/duanyuanchuan/Task/nnUnet_data/nnUNet_preprocessed/Task001_npc/nnUNetPlansv2.1_plans_3D.pkl'
path=r'/home/duanyuanchuan/projects/nnUNet_dataset/nnUNet_preprocessed/Task058_LEADCBAndBMSMonth7Data33/nnUNetPlansv2.1_plans_3D.pkl'
with (open(path, 'rb')) as f:
    s = pkl.load(f)
    print(s['plans_per_stage'][0]['batch_size'])
    print(s['plans_per_stage'][1]['batch_size'])
    # print(s['plans_per_stage'][0]['patch_size'])

    plans = load_pickle(path)
    plans['plans_per_stage'][0]['batch_size'] = 4
    plans['plans_per_stage'][1]['batch_size'] = 4
    # plans['plans_per_stage'][0]['patch_size'] = np.array((28, 192, 192))

    save_pickle(plans, join(r'/home/duanyuanchuan/projects/nnUNet_dataset/nnUNet_preprocessed/Task058_LEADCBAndBMSMonth7Data33/nnUNetPlansv2.1_DECETV_plans_3D.pkl'))  # 路径的保存必须以_plans_xD.pkl结尾才能被识别
