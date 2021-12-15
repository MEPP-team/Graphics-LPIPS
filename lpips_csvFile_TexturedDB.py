import argparse
import os
import lpips
import torch
import numpy as np
import statsmodels.api as sm
from scipy import stats
import csv
from itertools import groupby
from operator import itemgetter
from statistics import mean
from decimal import Decimal


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f','--csvfile', type=str, default='./dataset/TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv')
parser.add_argument('-m','--modelpath', type=str, default='./lpips/weights/v0.1/alex.pth', help='location of model')
parser.add_argument('-o','--out', type=str, default='./GraphicsLPIPS_TestsetScores.csv')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

root_refPatches = './dataset\\References_patches_withVP_threth0.6'
root_distPatches = './dataset\\PlaylistsStimuli_patches_withVP_threth0.6'

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version, model_path = opt.modelpath)# e.g. model_path = './checkpoints/Trial1/latest_net_.pth'
if(opt.use_gpu):
	loss_fn.cuda()
    
## Output file
f = open(opt.out,'w')
f.writelines('p0,lpips_alex,MOS\n')

## read Input csv file 
List_MOS = []
List_GraphicsLPIPS= []

with open(opt.csvfile) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            dist = row[1]
            model = row[0]
            MOS = row[2]
            nbPatches = int(row[3])# for VP1
            
            res = []
            resString =''
            for p in range(1, nbPatches +1):
                refpatch = model + '_Ref_P' + str(p) + '.png'
                refpath = os.path.join(root_refPatches, refpatch)
                stimuluspatch = dist + '_P' + str(p) + '.png'
                stimuluspath = os.path.join(root_distPatches, stimuluspatch)
                    
                img0 = lpips.im2tensor(lpips.load_image(refpath)) # RGB image from [-1,1]
                img1 = lpips.im2tensor(lpips.load_image(stimuluspath))
                
                if(opt.use_gpu):
                    img0 = img0.cuda()
                    img1 = img1.cuda()
                    
                dist01 = loss_fn.forward(img0,img1).reshape(1,).item()
                if dist01 > 1:
                    dist01 = 1
                res.append(dist01)
    
            Graphicslpips = sum(res)/len(res)
            List_GraphicsLPIPS.append(Graphicslpips)
            List_MOS.append(float(MOS))
            f.writelines('%s, %.6f, %s\n'%(dist,Graphicslpips,MOS))

f.close()
List_GraphicsLPIPS = np.array(List_GraphicsLPIPS)
List_MOS = np.array(List_MOS)

# Instantiate a binomial family model with the logit link function (the default link function).
List_GraphicsLPIPS = sm.add_constant(List_GraphicsLPIPS)
glm_binom = sm.GLM(List_MOS, List_GraphicsLPIPS, family = sm.families.Binomial())#, link = sm.families.links.Logit()
res_regModel = glm_binom.fit()

fitted_GraphicsLpips = res_regModel.predict()
corrPears =  stats.pearsonr(fitted_GraphicsLpips, List_MOS)[0]
corrSpear =  stats.spearmanr(fitted_GraphicsLpips, List_MOS)[0]
print('pearson %.3f'%corrPears)
print('spearman %.3f'%corrSpear)


