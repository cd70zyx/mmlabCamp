# 作者 : 钢之乐学 - 西电
# 功能 : 按各个类别划分数据集train-val

import os
import os.path as osp
import shutil

BASEDIR = osp.dirname(osp.abspath('.'))

DatasetDIR = 'data/flower_dataset'
SplitDataDir = 'data/flower_dataset_split'

trainDIR = osp.join(SplitDataDir, 'train')
valDIR = osp.join(SplitDataDir, 'val')

clsnames = os.listdir(DatasetDIR)
for clsname in clsnames:
    clsdir = osp.join(DatasetDIR, clsname)
    train_clsdir = osp.join(trainDIR, clsname)
    val_clsdir = osp.join(valDIR, clsname)
    if not osp.exists(train_clsdir): os.makedirs(train_clsdir)
    if not osp.exists(val_clsdir): os.makedirs(val_clsdir)
    samplenames = os.listdir(clsdir)
    num_samples = len(samplenames)
    for i, name in enumerate(samplenames):
        src_path = osp.join(clsdir, name)
        if i < num_samples * 0.8:
            dst_path = osp.join(train_clsdir, name)
        else:
            dst_path = osp.join(val_clsdir, name)    
        shutil.copy(src_path, dst_path)
    print(clsname, 'copied')
