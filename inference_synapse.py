import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy import metric

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())
def hd(pred,gt):
        if pred.sum() > 0 and gt.sum()>0:
            hd95 = metric.binary.hd95(pred, gt)
            return  hd95
        else:
            return 0
            
def process_label(label):
    ABM = label == 1

   
    return ABM

def test(fold):
    path='/mnt/e/htnnunet+/nnUNet-master/DATASET/predict_tumor'
    label_list=sorted(glob.glob(os.path.join(path,'labelTs','*nii.gz')))
    infer_list=sorted(glob.glob(os.path.join(path,'inferTs',fold,'*nii.gz')))
    print("loading success...")
    print(label_list)
    print(infer_list)
    Dice_ABM=[]

    
    hd_ABM=[]

    
    file=path + 'inferTs/'+fold
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file+'/dice_pre.txt', 'a')
    
    for label_path,infer_path in zip(label_list,infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label,infer = read_nii(label_path),read_nii(infer_path)
        label_ABM=process_label(label)
        infer_ABM=process_label(infer)
        
        Dice_ABM.append(dice(infer_ABM,label_ABM))

        
        hd_ABM.append(hd(infer_ABM,label_ABM))

        
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        fw.write('Dice_ABM: {:.4f}\n'.format(Dice_ABM[-1]))

        
        fw.write('hd_ABM: {:.4f}\n'.format(hd_ABM[-1]))

        
        dsc=[]
        HD=[]
        dsc.append(Dice_ABM[-1])

        fw.write('DSC:'+str(np.mean(dsc))+'\n')
        
        HD.append(hd_ABM[-1])

        fw.write('hd:'+str(np.mean(HD))+'\n')
        
    
    fw.write('*'*20+'\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_ABM'+str(np.mean(Dice_ABM))+'\n')

    
    fw.write('Mean_hd\n')
    fw.write('hd_ABM'+str(np.mean(hd_ABM))+'\n')

   
    fw.write('*'*20+'\n')
    
    dsc=[]
    dsc.append(np.mean(Dice_ABM))

    fw.write('dsc:'+str(np.mean(dsc))+'\n')
    
    HD=[]
    HD.append(np.mean(hd_ABM))

    fw.write('hd:'+str(np.mean(HD))+'\n')
    
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", help="fold name")
    args = parser.parse_args()
    fold=args.fold
    test(fold)
